import airsim
import math
import numpy as np
import re
import cv2
 
objects_dict = {
    "turbine1": "BP_Wind_Turbines_C_1",
    "turbine2": "StaticMeshActor_2",
    "solarpanels": "StaticMeshActor_146",
    "crowd": "StaticMeshActor_6",
    "car": "StaticMeshActor_10",
    "tower1": "SM_Electric_trellis_179",
    "tower2": "SM_Electric_trellis_7",
    "tower3": "SM_Electric_trellis_8",
}
 
 
class AirSimWrapper:
    def __init__(self):
        self.client = airsim.MultirotorClient(ip="134.88.13.90") # 134.88.13.90
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
 
    def takeoff(self):
        self.client.takeoffAsync().join()
 
    def land(self):
        self.client.landAsync(timeout_sec=300).join()
 
    def go_home(self):
        self.client.goHomeAsync().join()
   
    def land_status(self):
        landed = self.client.getMultirotorState().landed_state
        if landed == 0:
            return True
        else:
            return False
 
    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]
   
    def get_state(self):
        pose = self.client.simGetVehiclePose()
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        yaw = math.degrees(yaw)
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val, yaw]
    
    def fly_to(self, point, speed):
        self.client.moveToPositionAsync(point[0], point[1], point[2], speed, 20).join()
   
    def move_velocity_body(self, v , t):
        self.client.moveByVelocityBodyFrameAsync(v[0], v[1], v[2], t).join()

    def hover(self):
        self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 1).join()
       
    def fly_path(self, points):
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 10, 1).join()
 
    def set_yaw(self, yaw):
        self.client.rotateToYawAsync(yaw, 5).join()
 
    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return (math.degrees(yaw) + 360) % 360
   
    def rotate_yaw(self, degree):
        # radian = math.radians(degree)
        self.client.rotateByYawRateAsync(degree, 1).join()
           
    def get_position(self, object_name):
        query_string = objects_dict[object_name] + ".*"
        object_names_ue = []
        while len(object_names_ue) == 0:
            object_names_ue = self.client.simListSceneObjects(query_string)
        pose = self.client.simGetObjectPose(object_names_ue[0])
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]
   
    def move_velocity(self, vx, vy, vz):
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.2).join()
       
    def get_distance(self):
        return self.client.getDistanceSensorData().distance
   
    def get_image(self):
        png_image = self.client.simGetImage("0", airsim.ImageType.Scene)
        png_image = cv2.imdecode(airsim.string_to_uint8_array(png_image), cv2.IMREAD_UNCHANGED)
        return png_image[: ,: ,0:3]
   
    def get_depth_image(self):
        # png_depth_image = self.client.simGetImage("0", airsim.ImageType.DepthVis)
        # png_depth_image = cv2.imdecode(airsim.string_to_uint8_array(png_depth_image), cv2.IMREAD_UNCHANGED)
        # png_depth_image = cv2.cvtColor(png_depth_image, cv2.COLOR_BGR2GRAY)
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)])
        response = responses[0]
 
        # get numpy array
        img1d = response.image_data_float
 
        # reshape array to 2D array H X W
        png_depth_image = np.reshape(img1d,(response.height, response.width))
        return png_depth_image/100
   
    def turn_left(self):
        self.set_yaw(self.get_yaw()-90)
       
    def turn_right(self):
        self.set_yaw(self.get_yaw()+90)
       
    def drone_control(self, move):
        pass
       
    def reset_airsim(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
   
    def get_collision(self):
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        else:
            return False
        
    #ry added
    # 在AirSimWrapper类中添加以下方法：
    def check_obstacle_distance(self, x, y, z):
        """检查到最近障碍物的距离"""
        # 使用深度图像获取障碍物距离
        depth_image = self.get_depth_image()
        min_distance = np.min(depth_image) * 100  # 转换为厘米
        return min_distance

    def get_velocity(self):
        """获取当前速度"""
        state = self.client.getMultirotorState()
        return [
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val
        ]