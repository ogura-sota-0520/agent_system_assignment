#!/usr/bin/env python3
"""
PIDコントローラーのテストプログラム
- 2値画像の表示
- 1本の白線の中心の表示
- Twistメッセージの表示
"""

import sys
import os
import cv2

# PIDControllerクラスをインポートするためのパス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ROSを使わないモックを作成


class MockRospy:
    def init_node(self, name, anonymous=True):
        print(f"Mock: ROS node '{name}' initialized")

    def spin(self):
        print("Mock: ROS spin called")

    def Subscriber(self, topic, msg_type, callback):
        print(f"Mock: Subscribed to {topic}")
        return MockSubscriber(topic, msg_type, callback)

    def Publisher(self, topic, msg_type, queue_size=10):
        print(f"Mock: Publisher created for {topic}")
        return MockPublisher(topic, msg_type, queue_size)

    class ROSInterruptException(Exception):
        pass


class MockImage:
    def __init__(self, cv_image):
        self.cv_image = cv_image


class MockTwist:
    def __init__(self):
        self.linear = type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0})()
        self.angular = type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0})()


class MockSubscriber:
    def __init__(self, topic, msg_type, callback):
        print(f"Mock: Subscribed to {topic}")
        self.callback = callback


class MockPublisher:
    def __init__(self, topic, msg_type, queue_size=10):
        print(f"Mock: Publisher created for {topic}")

    def publish(self, msg):
        print(f"Mock: Published message - linear.x: {msg.linear.x}, angular.z: {msg.angular.z}")


class MockCvBridge:
    def imgmsg_to_cv2(self, msg, encoding):
        return msg.cv_image


# ROSモジュールをモックで置き換え
sys.modules['rospy'] = MockRospy()
sys.modules['sensor_msgs.msg'] = type('module', (), {'Image': MockImage})()
sys.modules['geometry_msgs.msg'] = type('module', (), {'Twist': MockTwist})()
sys.modules['cv_bridge'] = type('module', (), {'CvBridge': MockCvBridge})()

# PIDControllerをインポート
from pid_controller import PIDController


class PIDControllerTester:
    def __init__(self):
        """テスト用のPIDコントローラー"""
        # 実際のPIDControllerを作成（ROSモックを使用）
        self.controller = PIDController()

    def run_test(self, image_path):
        """テスト実行"""
        print("=== PID Controller Test ===")
        print(f"Test image: {image_path}")

        # 画像読み込み
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return

        print(f"Image size: {image.shape[1]} x {image.shape[0]}")

        # 1. 2値画像に変換して表示
        binary_image = self.controller.process_image(image)
        print("✓ Binary image conversion completed")
        print(f"Binary image shape: {binary_image.shape}")
        print(f"Binary image pixels (white): {cv2.countNonZero(binary_image)}")
        print(f"Binary image pixels (black): {binary_image.size - cv2.countNonZero(binary_image)}")
        print(f"Binary image min/max values: {binary_image.min()}/{binary_image.max()}")

        # 2値画像をファイルに保存（表示の代わり）
        binary_output_path = os.path.join(os.path.dirname(image_path), 'binary_output.png')
        cv2.imwrite(binary_output_path, binary_image)
        print(f"Binary image saved to: {binary_output_path}")

        # グレースケール画像も保存してデバッグ
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_output_path = os.path.join(os.path.dirname(image_path), 'gray_output.png')
        cv2.imwrite(gray_output_path, gray)
        print(f"Grayscale image saved to: {gray_output_path}")
        print(f"Grayscale image min/max values: {gray.min()}/{gray.max()}")

        # 2. 白線の中心計算と表示
        error = self.controller.calculate_error(binary_image)

        # 可視化のための詳細情報を取得
        height, width = binary_image.shape
        mid_x = width // 2

        # 輪郭検出
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 1:
            # 面積が最大の輪郭を選択
            largest_contour = max(contours, key=cv2.contourArea)

            # 重心計算
            moment = cv2.moments(largest_contour)

            if moment['m00'] != 0:
                line_center_x = int(moment['m10'] / moment['m00'])

                print("✓ White line center detected:")
                print(f"  Line center: x={line_center_x}")
                print(f"  Image center: x={mid_x}")
                print(f"  Error: {error} pixels")

                # 白線中心を可視化
                vis_image = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)

                # 画像中心線（緑）
                cv2.line(vis_image, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)

                # 白線の中心点（青）
                cv2.circle(vis_image, (line_center_x, height // 2), 8, (255, 0, 0), -1)

                # 白線の中心線（赤）
                cv2.line(vis_image, (line_center_x, 0), (line_center_x, height), (0, 0, 255), 2)

                cv2.putText(vis_image, f"Error: {error}px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 可視化画像をファイルに保存（表示の代わり）
                vis_output_path = os.path.join(os.path.dirname(image_path), 'line_center_output.png')
                cv2.imwrite(vis_output_path, vis_image)
                print(f"White line center visualization saved to: {vis_output_path}")
            else:
                print("✗ Could not calculate line center")
        else:
            print("✗ Could not detect white line")

        # 3. PID制御とTwist表示
        twist = self.controller.pid_control(error)
        print("✓ PID Control output (Twist message):")
        print("  geometry_msgs/Twist:")
        print("    linear:")
        print(f"      x: {twist.linear.x:.3f}")
        print(f"      y: {twist.linear.y:.3f}")
        print(f"      z: {twist.linear.z:.3f}")
        print("    angular:")
        print(f"      x: {twist.angular.x:.3f}")
        print(f"      y: {twist.angular.y:.3f}")
        print(f"      z: {twist.angular.z:.6f}")

        print("\nTest completed!")


def main():
    """メイン関数"""
    # テスト画像のパス
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'data', 'image.png')

    # テスト実行
    tester = PIDControllerTester()
    tester.run_test(image_path)


if __name__ == '__main__':
    main()
