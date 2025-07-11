"""
1. /go2_description/camera1/imageトピックから画像を取得
2. 画像上の１本の白線を検出
3. １本の白線の中心を計算
4. １本の白線の中心と画像の中心の差分を計算
5. 差分を基にPID制御を行い、ロボットの移動方向を/cmd_velをパブリッシュすることで調整
"""
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2


class PIDController:
    def __init__(self):
        rospy.init_node('pid_controller', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/go2_description/camera1/image', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # PID parameters
        self.kp = 0.01  # Proportional gain
        self.ki = 0.0005  # Integral gain
        self.kd = 0.0025  # Derivative gain

        self.prev_error = 0
        self.integral = 0

    def image_callback(self, msg):
        """
        カメラ画像を受信してロボットの制御命令を生成するコールバック関数

        処理の流れ:
        1. ROS Image メッセージを OpenCV 形式に変換
        2. 画像を処理して白線を検出しやすい形に変換
        3. 1本の白線の中心位置と画像中心の誤差を計算
        4. PID制御により誤差を補正する制御信号を生成
        5. 生成した制御信号をロボットに送信
        """
        # ROS Image メッセージを OpenCV の BGR8 形式に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 画像を処理して白線検出用の二値画像に変換
        processed_image = self.process_image(cv_image)

        # 1本の白線の中心位置と画像中心の誤差を計算
        error = self.calculate_error(processed_image)

        # PID制御により誤差を補正する制御信号（Twist）を生成
        twist = self.pid_control(error)

        # 生成した制御信号をロボットに送信
        self.cmd_vel_pub.publish(twist)

    def process_image(self, image):
        """
        カメラ画像を白線検出用の二値画像に変換する関数

        Args:
            image: OpenCV形式のBGR画像

        Returns:
            thresh: 白線が白(255)、背景が黒(0)の二値画像

        処理内容:
        1. カラー画像をグレースケールに変換
        2. 閾値処理により白線部分を抽出（明度200以上を白線として検出）
        """
        # カラー画像をグレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 閾値処理：明度200以上を白線として検出（白線を白く表示）
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return thresh

    def calculate_error(self, image):
        """
        二値画像から1本の白線を検出し、画像中心との誤差を計算する関数

        Args:
            image: 白線検出済みの二値画像

        Returns:
            error: 画像中心と白線中心のX座標差分（ピクセル単位）
                  正の値：ロボットが左に寄っている（右に曲がる必要あり）
                  負の値：ロボットが右に寄っている（左に曲がる必要あり）

        処理内容:
        1. 輪郭検出により白線を抽出
        2. 面積が最大の輪郭を白線として選択
        3. 白線の重心を計算
        4. 白線の中心と画像中心の差分を誤差として算出
        """
        height, width = image.shape
        mid_x = width // 2

        # 輪郭検出により白線を抽出
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 白線が検出されない場合は誤差0を返す
        if len(contours) < 1:
            return 0

        # 面積が最大の輪郭を白線として選択
        largest_contour = max(contours, key=cv2.contourArea)

        # 輪郭の重心（モーメント）を計算
        moment = cv2.moments(largest_contour)

        # 重心計算に失敗した場合（面積が0の場合）は誤差0を返す
        if moment['m00'] == 0:
            return 0

        # 白線の重心のX座標を計算
        line_center_x = int(moment['m10'] / moment['m00'])

        # 画像中心と白線中心の差分を誤差として計算
        error = mid_x - line_center_x
        return error

    def pid_control(self, error):
        """
        PID制御により誤差を補正する制御信号を生成する関数

        Args:
            error: 画像中心と白線中心の誤差（ピクセル単位）

        Returns:
            twist: ロボットの移動制御信号（geometry_msgs/Twist）
                  linear.x: 前進速度（固定値0.2）
                  angular.z: 旋回速度（PID制御で計算）

        PID制御の構成要素:
        - P制御（比例）: 現在の誤差に比例した補正
        - I制御（積分）: 過去の誤差の累積に基づく補正（定常偏差の除去）
        - D制御（微分）: 誤差の変化率に基づく補正（オーバーシュートの抑制）
        """
        # D制御: 誤差の変化率を計算
        derivative = error - self.prev_error

        # I制御:誤差の積分値を更新
        self.integral += error

        # PID制御出力を計算（比例 + 積分 + 微分）
        control_output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Twist メッセージを作成
        twist = Twist()
        # 旋回速度を[-0.3, 0.3]の範囲に制限
        twist.linear.y = max(-0.06, min(0.06, control_output))  # 前進速度は固定値0.2
        angular_z = control_output  # 負号で方向を調整
        twist.angular.z = max(-0.3, min(0.3, angular_z))

        # 次回計算のために現在の誤差を保存
        self.prev_error = error

        return twist

    def run(self):
        """
        PIDコントローラのメインループ
        ROSのスピンを開始してコールバック関数の実行を待機
        """
        rospy.spin()


if __name__ == '__main__':
    try:
        # PIDコントローラのインスタンスを作成
        controller = PIDController()
        print("PID Controller started. Waiting for camera images...")

        # メインループを開始
        controller.run()

    except rospy.ROSInterruptException:
        print("PID Controller interrupted.")
    except Exception as e:
        print(f"Error: {e}")
