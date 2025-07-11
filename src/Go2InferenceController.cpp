#include <cnoid/MessageView>
#include <cnoid/SimpleController>
#include <cnoid/ValueTree>
#include <cnoid/YAMLReader>
#include <cnoid/EigenUtil>

#include <torch/torch.h>
#include <torch/script.h>
#include <geometry_msgs/Twist.h>
#include <ros/node_handle.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random> // C++標準乱数
#include <mutex>

using namespace cnoid;
namespace fs = std::filesystem;

class Go2InferenceController : public SimpleController
{
    Body *ioBody;
    double dt;
    double inference_dt = 0.02; // 推論の実行周期
    size_t inference_interval_steps;

    Vector3 global_gravity; // 長さ3のベクトルはVecto3を使う
    VectorXd last_action;   // 長さ不定のベクトルはVectorXdを使う
    VectorXd default_dof_pos;
    VectorXd target_dof_pos;
    // VectorXd target_dof_pos_prev;
    // VectorXd target_dof_vel;
    std::vector<std::string> motor_dof_names;

    torch::jit::script::Module model; // TorchScript形式を読み込むモデル

    // Config values
    double P_gain;
    double D_gain;
    int num_actions;
    double action_scale;
    double ang_vel_scale;
    double lin_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    Vector3 command_scale;

    // Command resampling
    Vector3d command;
    Vector2d lin_vel_x_range;
    Vector2d lin_vel_y_range;
    Vector2d ang_vel_range;
    size_t resample_interval_steps;
    size_t step_count = 0;

    // 安全機能用変数
    bool emergency_stop = false;
    Vector3d previous_position = Vector3d::Zero();
    double max_tilt_angle = 0.5; // 最大傾斜角度（ラジアン）

    // 乱数生成器
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist_lin_x;
    std::uniform_real_distribution<double> dist_lin_y;
    std::uniform_real_distribution<double> dist_ang;

    // ros用
    std::unique_ptr<ros::NodeHandle> node;
    ros::Subscriber subscriber;
    geometry_msgs::Twist latest_command_velocity;
    std::mutex command_velocity_mutex;

public:
    // コントローラがプロジェクトに読み込まれたときに実行される
    virtual bool configure(SimpleControllerConfig *config) override
    {
        node.reset(new ros::NodeHandle);
        return true;
    }
    // モデルの読み込み，各種設定値の読み込み，シミュレーションの初期化
    // シミュレーション開始の直前に実行される
    virtual bool initialize(SimpleControllerIO *io) override
    {
        dt = io->timeStep();
        ioBody = io->body();

        // ＜推論の実行周期 =0.02[s]＞/＜コントローラ（PD制御）と物理演算の実行周期 =0.001[s]＞
        inference_interval_steps = static_cast<int>(std::round(inference_dt / dt));
        std::ostringstream oss;
        oss << "inference_interval_steps: " << inference_interval_steps;
        MessageView::instance()->putln(oss.str());

        global_gravity = Vector3(0.0, 0.0, -1.0);

        // 関節の駆動モード，入出力情報の設定
        for (auto joint : ioBody->joints())
        {
            joint->setActuationMode(JointTorque);
            io->enableOutput(joint, JointTorque);
            io->enableInput(joint, JointAngle | JointVelocity); // 関節角度と速度の取得
        }
        // ルートリンクの位置・姿勢(Position)と速度・角速度(Twist)を取得する
        io->enableInput(ioBody->rootLink(), LinkPosition | LinkTwist);

        // commandの初期値
        command = Vector3d(0.0, 0.0, 0.0);

        // find the cfgs file
        // 設定ファイルcfgs.yamlの読み込み
        // ${HOME}/genesis_ws/logs/go2-walking/inference_tutorial/cfgs.yaml
        fs::path inference_target_path = fs::path(std::getenv("HOME")) / fs::path("genesis_ws/logs/go2-walking/sub2_com_fric0.2-1.8_kp18-30_kv0.7-1.2_rotI0.01-0.15_iter200");
        // fs::path inference_target_path = fs::path(std::getenv("HOME")) / fs::path("genesis_ws/logs/go2-walking/sub2_com_iter300");
        fs::path cfgs_path = inference_target_path / fs::path("cfgs.yaml");
        if (!fs::exists(cfgs_path))
        {
            oss << cfgs_path << " is not found!!!";
            MessageView::instance()->putln(oss.str());
            return false;
        }

        // load configs
        YAMLReader reader; // YAMLReaderによる読み込み
        auto root = reader.loadDocument(cfgs_path)->toMapping();

        auto env_cfg = root->findMapping("env_cfg");
        auto obs_cfg = root->findMapping("obs_cfg");
        auto command_cfg = root->findMapping("command_cfg");

        // env_cfg
        P_gain = env_cfg->get("kp", 24);
        D_gain = env_cfg->get("kd", 0.95);

        // TODO:
        P_gain = 35;
        D_gain = 1.7;

        resample_interval_steps = static_cast<int>(std::round(env_cfg->get("resampling_time_s", 4.0) / dt));

        // joint values
        num_actions = env_cfg->get("num_actions", 1);
        last_action = VectorXd::Zero(num_actions);
        default_dof_pos = VectorXd::Zero(num_actions);

        // ロボットモデルの関節順序と，action・obsベクトルの関節順序の対応の設定
        // env_cfgのdof_namesのリストの順にaction・obsベクトルを構成する
        auto dof_names = env_cfg->findListing("dof_names");
        motor_dof_names.clear();
        for (int i = 0; i < dof_names->size(); ++i)
        {
            motor_dof_names.push_back(dof_names->at(i)->toString());
        }

        // default_angles(関節角度の-1~1へ標準化の中央値)
        // default_anglesもactionベクトルの並び
        auto default_angles = env_cfg->findMapping("default_joint_angles");
        for (int i = 0; i < motor_dof_names.size(); ++i)
        {
            std::string name = motor_dof_names[i];
            default_dof_pos[i] = default_angles->get(name, 0.0);
        }

        // その他のscalesやrangeパラメータの読み込み，コマンド用乱数の初期化
        // use default_dof_pos for initializing target angles
        target_dof_pos = default_dof_pos;
        // target_dof_pos_prev = default_dof_pos;
        // target_dof_vel = VectorXd::Zero(num_actions);

        // scales
        action_scale = env_cfg->get("action_scale", 1.0);

        // obs_cfg
        ang_vel_scale = obs_cfg->findMapping("obs_scales")->get("ang_vel", 1.0);
        lin_vel_scale = obs_cfg->findMapping("obs_scales")->get("lin_vel", 1.0);
        dof_pos_scale = obs_cfg->findMapping("obs_scales")->get("dof_pos", 1.0);
        dof_vel_scale = obs_cfg->findMapping("obs_scales")->get("dof_vel", 1.0);

        command_scale[0] = lin_vel_scale;
        command_scale[1] = lin_vel_scale;
        command_scale[2] = ang_vel_scale;

        // command_cfg
        auto range_listing = command_cfg->findListing("lin_vel_x_range");
        lin_vel_x_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());
        range_listing = command_cfg->findListing("lin_vel_y_range");
        lin_vel_y_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());
        range_listing = command_cfg->findListing("ang_vel_range");
        ang_vel_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());

        // モータDOFのindex取得

        // 乱数初期化
        rng.seed(std::random_device{}());
        dist_lin_x = std::uniform_real_distribution<double>(lin_vel_x_range[0], lin_vel_x_range[1]);
        dist_lin_y = std::uniform_real_distribution<double>(lin_vel_y_range[0], lin_vel_y_range[1]);
        dist_ang = std::uniform_real_distribution<double>(ang_vel_range[0], ang_vel_range[1]);

        // load the network model
        // TorchScript形式のネットワークの読み込み (policy_traced.pt)
        fs::path model_path = inference_target_path / fs::path("policy_traced.pt");
        if (!fs::exists(model_path))
        {
            oss << model_path << " is not found!!!";
            MessageView::instance()->putln(oss.str());
            return false;
        }
        // model = torch::jit::load(model_path); // CUDA
        // model.to(torch::kCUDA);
        model = torch::jit::load(model_path, torch::kCPU); // CPU形式で読み込む
        model.to(torch::kCPU);
        model.eval();

        // コールバック関数の登録
        subscriber = node->subscribe("cmd_vel", 1, &Go2InferenceController::command_velocity_callback, this);

        return true;
    }
    // コールバック関数の実装
    void command_velocity_callback(const geometry_msgs::Twist &msg)
    {
        std::lock_guard<std::mutex> lock(command_velocity_mutex);
        latest_command_velocity = msg;
    }
    // 推論処理を行う
    bool inference(VectorXd &target_dof_pos, const Vector3d &angular_velocity, const Vector3d &projected_gravity, const VectorXd &joint_pos, const VectorXd &joint_vel)
    {
        try
        {
            // 観測値のベクトルの作成 env.step()と同様にスケール，-1~1へ標準化
            // observation vector
            std::vector<float> obs_vec;
            for (int i = 0; i < 3; ++i)
                obs_vec.push_back(angular_velocity[i] * ang_vel_scale);
            for (int i = 0; i < 3; ++i)
                obs_vec.push_back(projected_gravity[i]);
            for (int i = 0; i < 3; ++i)
                obs_vec.push_back(command[i] * command_scale[i]);
            for (int i = 0; i < num_actions; ++i)
                obs_vec.push_back((joint_pos[i] - default_dof_pos[i]) * dof_pos_scale);
            for (int i = 0; i < num_actions; ++i)
                obs_vec.push_back(joint_vel[i] * dof_vel_scale);
            for (int i = 0; i < num_actions; ++i)
                obs_vec.push_back(last_action[i]);

            // auto input = torch::from_blob(obs_vec.data(), {1, (long)obs_vec.size()}, torch::kFloat32).to(torch::kCUDA);
            // std::vectorからtorch::Tensorを作成する
            // blobとはBinary Large Objectの略で，連続したバイト列を表す汎用的なデータ構造
            auto input = torch::from_blob(obs_vec.data(), {1, (long)obs_vec.size()}, torch::kFloat32).to(torch::kCPU);

            // IvalueはTensor,int,float,bool,tuple,list,dict等の汎用型．汎用性の為にIvalueのvectorにする
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);

            // inference
            // 推論処理
            torch::Tensor output = model.forward(inputs).toTensor();
            auto output_cpu = output.to(torch::kCPU);
            auto output_acc = output_cpu.accessor<float, 2>();

            VectorXd action(num_actions);
            for (int i = 0; i < num_actions; ++i)
            {
                last_action[i] = output_acc[0][i];
                action[i] = last_action[i];
            }

            target_dof_pos = action * action_scale + default_dof_pos; // actionから関節角度への変換
        }
        catch (const c10::Error &e)
        {
            std::cerr << "Inference error: " << e.what() << std::endl;
        }

        return true;
    }

    // 周期実行する関数(choreonoidのシミュレーションのdt=0.001[s]毎)
    // go2_eval.pyのwhile文内の
    // actions = policy(obs)
    // obs, _, rews, dones, infos = env.step(actions)
    // に相当する
    virtual bool control() override
    {
        geometry_msgs::Twist command_velocity;
        {
            std::lock_guard<std::mutex> lock(command_velocity_mutex);
            command_velocity = latest_command_velocity;
        }

        // commandの値を段階的に増加（安定性重視）
        // command[0] = command_velocity.linear.x;
        static double target_speed = 0.1; // 初期速度を低く設定

        // TODO: 最大速度の調整
        const double max_speed = 0.5; // より安定な最大速度

        // 1秒(1000ステップ)毎に速度を少しずつ上げる
        if (step_count > 0 && step_count % 1000 == 0 && target_speed < max_speed)
        {
            target_speed += 0.05;                             // 0.05m/sずつ増加
            target_speed = std::min(target_speed, max_speed); // 最大速度を超えないように
        }

        command[0] = target_speed;
        command[1] = command_velocity.linear.y;
        command[2] = command_velocity.angular.z;

        // 安全機能：姿勢監視
        const auto rootLink = ioBody->rootLink();
        const Isometry3d root_coord = rootLink->T();

        // 傾斜角度チェック（ロール・ピッチ）
        Vector3d rpy = rpyFromRot(root_coord.linear());
        if (abs(rpy[0]) > max_tilt_angle || abs(rpy[1]) > max_tilt_angle)
        {
            emergency_stop = true;
            target_speed = 0.0;
            std::cout << "Emergency stop: Excessive tilt detected! Roll: " << rpy[0] << ", Pitch: " << rpy[1] << std::endl;
        }

        // 高さチェック
        double current_height = root_coord.translation().z();
        if (current_height < 0.15)
        { // 地面から15cm以下
            emergency_stop = true;
            target_speed = 0.0;
            std::cout << "Emergency stop: Robot too low! Height: " << current_height << std::endl;
        }

        // 緊急停止時は速度を0に
        if (emergency_stop)
        {
            command[0] = 0.0;
            command[1] = 0.0;
            command[2] = 0.0;
        }

        std::cout << "command velocity:" << command.transpose() << std::endl;

        // get current states
        // 現在状態を取得し，観測値（モデルへの入力できる値）に変換する
        // env.step()と同様に，胴体の角速度と重力を胴体リンク座標系相対の値に変換する
        Vector3 angular_velocity = root_coord.linear().transpose() * rootLink->w();
        Vector3 projected_gravity = root_coord.linear().transpose() * global_gravity;

        // 関節角度，角速度の取得
        // obsベクトルの順序に合わせるために，motor_dof_namesに従って関節を取得する
        VectorXd joint_pos(num_actions), joint_vel(num_actions);
        for (int i = 0; i < num_actions; ++i)
        {
            auto joint = ioBody->joint(motor_dof_names[i]);
            joint_pos[i] = joint->q();
            joint_vel[i] = joint->dq();
        }

        // inference
        if (step_count % inference_interval_steps == 0)
        {
            inference(target_dof_pos, angular_velocity, projected_gravity, joint_pos, joint_vel);
            // target_dof_vel = (target_dof_pos - target_dof_pos_prev) / inference_dt;
            // target_dof_pos_prev = target_dof_pos;
        }

        // set target outputs
        // 関節PD制御（action=target_dof_posの順序に合わせる）
        for (int i = 0; i < num_actions; ++i)
        {
            auto joint = ioBody->joint(motor_dof_names[i]);
            double q = joint->q();
            double dq = joint->dq();
            // genesisのPD制御に合わせる （D制御は目標速度=0のダンピング制御のみ）
            // double u = P_gain * (target_dof_pos[i] - q) + D_gain * (target_dof_vel[i] - dq);
            double u = P_gain * (target_dof_pos[i] - q) + D_gain * (-dq);
            joint->u() = u; // 関節トルクの設定
        }

        // step_countを進める (chorenoidの制御周期dt毎に+1する)
        ++step_count;

        // ros::spin();

        return true;
    }
    virtual void stop() override
    {
        subscriber.shutdown();
    }
};
CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(Go2InferenceController)
