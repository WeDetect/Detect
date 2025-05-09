#ifndef LIDAR_TF_POSITION_PUBLISHER_NODE_HPP
#define LIDAR_TF_POSITION_PUBLISHER_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Transform.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <thread>

// Include for dynamic parameter handling
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rcl_interfaces/msg/floating_point_range.hpp"

using namespace std;

class LidarTFPublisher : public rclcpp::Node
{
//--------------------------------public--------------------------------
public:
    // Constructor
    LidarTFPublisher();

//--------------------------------private--------------------------------
private:
    // Publishers and broadcasters
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr lidar_pose_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Parameter callback handle
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
    
    // Parameter storage
    double tf_hz_;
    double lidar_x_;
    double lidar_y_;
    double lidar_z_;
    double lidar_roll_;
    double lidar_pitch_;
    double lidar_yaw_;
    std::string global_frame_;
    std::string lidar_frame_;
    
    // Messages
    geometry_msgs::msg::PoseStamped lidar_pose_;
    geometry_msgs::msg::TransformStamped lidar_transform_stamped_;

    // Methods for parameter handling
    void declare_dynamic_parameters();
    bool load_config(const std::string &file_path);
    void update_ros_parameters();
    void update_transform_and_pose();
    rcl_interfaces::msg::SetParametersResult parameters_callback(
        const std::vector<rclcpp::Parameter> &parameters);
    
    // Timer callback
    void timer_callback();
    
    // Parameter saving
    void save_parameters_to_yaml(const std::vector<std::string>& changed_params = {});
    
};

#endif // LIDAR_TF_POSITION_PUBLISHER_NODE_HPP
