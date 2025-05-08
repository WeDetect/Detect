#include "lidar_tf_position_publisher/lidar_tf_position_publisher_node.hpp" 
using namespace std;

// Constructor implementation
LidarTFPublisher::LidarTFPublisher() : Node("lidar_tf_position_publisher_node")
{   
    // Declare parameters with descriptors for dynamic reconfiguration
    declare_dynamic_parameters();
    
    // Get parameters from ROS parameter server (these will be loaded from the YAML file by the launch)
    tf_hz_ = this->get_parameter("tf_hz").as_double();
    lidar_x_ = this->get_parameter("lidar_x").as_double();
    lidar_y_ = this->get_parameter("lidar_y").as_double();
    lidar_z_ = this->get_parameter("lidar_z").as_double();
    lidar_roll_ = this->get_parameter("lidar_roll").as_double();
    lidar_pitch_ = this->get_parameter("lidar_pitch").as_double();
    lidar_yaw_ = this->get_parameter("lidar_yaw").as_double();
    global_frame_ = this->get_parameter("global_frame").as_string();
    lidar_frame_ = this->get_parameter("lidar_frame").as_string();
    
    // Update transform and pose based on parameters
    update_transform_and_pose();
    
    // Set up parameter callback for dynamic reconfiguration
    params_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&LidarTFPublisher::parameters_callback, this, std::placeholders::_1));
    
    // Create a timer to publish at the given Hz
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000 / static_cast<int>(tf_hz_)), 
        std::bind(&LidarTFPublisher::timer_callback, this));

    // Create publishers
    std::string package_name = "lidar_tf_position_publisher";
    lidar_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(package_name+"/lidar_pose", 10);
    
    // Create a transform broadcaster
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    
    // Register shutdown callback to save parameters
    rclcpp::on_shutdown([this]() {
        //print in orange
        RCLCPP_INFO(this->get_logger(), "\033[1;33mShutting down, saving parameters...\033[0m");
        this->save_parameters_to_yaml();
    });

    //print in yellow
    RCLCPP_INFO(this->get_logger(), "===============================");
    RCLCPP_INFO(this->get_logger(), "\033[1;33mParameters: x=%.2f, y=%.2f, z=%.2f, roll=%.2f, pitch=%.2f, yaw=%.2f, hz=%.2f\033[0m",
                lidar_x_, lidar_y_, lidar_z_, lidar_roll_, lidar_pitch_, lidar_yaw_, tf_hz_);

    //print in yellow that the node is initialized
    RCLCPP_INFO(this->get_logger(), "\033[1;33mLidarTFPublisher Node initialized\033[0m");
    RCLCPP_INFO(this->get_logger(), "===============================");
}

// Declare all parameters with descriptors for dynamic reconfiguration
void LidarTFPublisher::declare_dynamic_parameters() {
    // Create parameter descriptor with range for tf_hz
    rcl_interfaces::msg::ParameterDescriptor tf_hz_desc;
    tf_hz_desc.description = "Frequency of TF publishing in Hz";
    rcl_interfaces::msg::FloatingPointRange tf_hz_range;
    tf_hz_range.from_value = 1.0;
    tf_hz_range.to_value = 100.0;
    tf_hz_range.step = 1.0;
    tf_hz_desc.floating_point_range.push_back(tf_hz_range);
    
    // Position parameters with sliders
    rcl_interfaces::msg::ParameterDescriptor pos_x_desc;
    pos_x_desc.description = "X position of lidar in meters";
    rcl_interfaces::msg::FloatingPointRange pos_x_range;
    pos_x_range.from_value = -100.0;
    pos_x_range.to_value = 100.0;
    pos_x_range.step = 0.1;
    pos_x_desc.floating_point_range.push_back(pos_x_range);
    
    rcl_interfaces::msg::ParameterDescriptor pos_y_desc;
    pos_y_desc.description = "Y position of lidar in meters";
    rcl_interfaces::msg::FloatingPointRange pos_y_range;
    pos_y_range.from_value = -100.0;
    pos_y_range.to_value = 100.0;
    pos_y_range.step = 0.1;
    pos_y_desc.floating_point_range.push_back(pos_y_range);
    
    rcl_interfaces::msg::ParameterDescriptor pos_z_desc;
    pos_z_desc.description = "Z position of lidar in meters";
    rcl_interfaces::msg::FloatingPointRange pos_z_range;
    pos_z_range.from_value = -50.0;
    pos_z_range.to_value = 50.0;
    pos_z_range.step = 0.1;
    pos_z_desc.floating_point_range.push_back(pos_z_range);
    
    // Rotation parameters with sliders (in radians)
    rcl_interfaces::msg::ParameterDescriptor roll_desc;
    roll_desc.description = "Roll angle of lidar in radians";
    rcl_interfaces::msg::FloatingPointRange roll_range;
    roll_range.from_value = -3.15;
    roll_range.to_value = 3.15;
    roll_range.step = 0.001;
    roll_desc.floating_point_range.push_back(roll_range);
    
    rcl_interfaces::msg::ParameterDescriptor pitch_desc;
    pitch_desc.description = "Pitch angle of lidar in radians";
    rcl_interfaces::msg::FloatingPointRange pitch_range;
    pitch_range.from_value = -3.15;
    pitch_range.to_value = 3.15;
    pitch_range.step = 0.001;
    pitch_desc.floating_point_range.push_back(pitch_range);
    
    rcl_interfaces::msg::ParameterDescriptor yaw_desc;
    yaw_desc.description = "Yaw angle of lidar in radians";
    rcl_interfaces::msg::FloatingPointRange yaw_range;
    yaw_range.from_value = -3.15;
    yaw_range.to_value = 3.15;
    yaw_range.step = 0.001;
    yaw_desc.floating_point_range.push_back(yaw_range);
    
    // Frame parameters
    rcl_interfaces::msg::ParameterDescriptor frame_desc;
    frame_desc.description = "Frame ID";
    
    // Declare all parameters with descriptors
    this->declare_parameter<double>("tf_hz", 1.0, tf_hz_desc);
    this->declare_parameter<double>("lidar_x", 0.0, pos_x_desc);
    this->declare_parameter<double>("lidar_y", 0.0, pos_y_desc);
    this->declare_parameter<double>("lidar_z", 0.0, pos_z_desc);
    this->declare_parameter<double>("lidar_roll", 0.0, roll_desc);
    this->declare_parameter<double>("lidar_pitch", 0.0, pitch_desc);
    this->declare_parameter<double>("lidar_yaw", 0.0, yaw_desc);
    this->declare_parameter<std::string>("global_frame", "world", frame_desc);
    this->declare_parameter<std::string>("lidar_frame", "innoviz", frame_desc);
}

// Update transform and pose based on current parameters
void LidarTFPublisher::update_transform_and_pose() {
    // Create the transform message
    lidar_transform_stamped_.header.stamp = this->now();
    lidar_transform_stamped_.header.frame_id = global_frame_;
    lidar_transform_stamped_.child_frame_id = lidar_frame_;
    
    // Set translation
    lidar_transform_stamped_.transform.translation.x = lidar_x_;
    lidar_transform_stamped_.transform.translation.y = lidar_y_;
    lidar_transform_stamped_.transform.translation.z = lidar_z_;
    
    // Set rotation using quaternion
    tf2::Quaternion quaternion;
    quaternion.setRPY(lidar_roll_, lidar_pitch_, lidar_yaw_);
    lidar_transform_stamped_.transform.rotation.x = quaternion[0];
    lidar_transform_stamped_.transform.rotation.y = quaternion[1];
    lidar_transform_stamped_.transform.rotation.z = quaternion[2];
    lidar_transform_stamped_.transform.rotation.w = quaternion[3];
    
    // Update pose message
    lidar_pose_.header.frame_id = global_frame_;
    lidar_pose_.pose.position.x = lidar_x_;
    lidar_pose_.pose.position.y = lidar_y_;
    lidar_pose_.pose.position.z = lidar_z_;
    lidar_pose_.pose.orientation.x = quaternion[0];
    lidar_pose_.pose.orientation.y = quaternion[1];
    lidar_pose_.pose.orientation.z = quaternion[2];
    lidar_pose_.pose.orientation.w = quaternion[3];
}

// Parameter callback for dynamic reconfiguration
rcl_interfaces::msg::SetParametersResult LidarTFPublisher::parameters_callback(
    const std::vector<rclcpp::Parameter> &parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";
    
    bool params_changed = false;
    
    for (const auto &param : parameters) {
        if (param.get_name() == "tf_hz") {
            tf_hz_ = param.as_double();
            // Recreate timer with new frequency
            timer_ = this->create_wall_timer(
                std::chrono::milliseconds(1000 / static_cast<int>(tf_hz_)), 
                std::bind(&LidarTFPublisher::timer_callback, this));
            RCLCPP_INFO(this->get_logger(), "Updated tf_hz: %.2f", tf_hz_);
            params_changed = true;
        } else if (param.get_name() == "lidar_x") {
            lidar_x_ = param.as_double();
            params_changed = true;
        } else if (param.get_name() == "lidar_y") {
            lidar_y_ = param.as_double();
            params_changed = true;
        } else if (param.get_name() == "lidar_z") {
            lidar_z_ = param.as_double();
            params_changed = true;
        } else if (param.get_name() == "lidar_roll") {
            lidar_roll_ = param.as_double();
            params_changed = true;
        } else if (param.get_name() == "lidar_pitch") {
            lidar_pitch_ = param.as_double();
            params_changed = true;
        } else if (param.get_name() == "lidar_yaw") {
            lidar_yaw_ = param.as_double();
            params_changed = true;
        } else if (param.get_name() == "global_frame") {
            global_frame_ = param.as_string();
            params_changed = true;
        } else if (param.get_name() == "lidar_frame") {
            lidar_frame_ = param.as_string();
            params_changed = true;
        }
    }
    
    // If any parameters changed, update the transform and pose
    if (params_changed) {
        update_transform_and_pose();
        RCLCPP_INFO(this->get_logger(), "Parameters updated");
        
        // Create a list of changed parameters
        std::vector<std::string> changed_param_names;
        for (const auto &param : parameters) {
            changed_param_names.push_back(param.get_name());
        }
        
        // Save parameters to file after significant changes
        save_parameters_to_yaml(changed_param_names);
    }
    
    return result;
}

void LidarTFPublisher::timer_callback()
{
    try {
        // Update timestamp
        lidar_transform_stamped_.header.stamp = this->now();
        lidar_pose_.header.stamp = this->now();
        
        // Broadcast the transform
        tf_broadcaster_->sendTransform(lidar_transform_stamped_);
        
        // Publish the PoseStamped message
        lidar_pose_pub_->publish(lidar_pose_);
    }
    catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception caught: %s", e.what());
    }
    catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception caught in timer callback");
    }
}

// Save parameters to YAML files
void LidarTFPublisher::save_parameters_to_yaml(const std::vector<std::string>& changed_params) {
    try {
        // Helper function to format doubles with decimal point
        auto format_double = [](double value) -> std::string {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << value;
            return oss.str();
        };
        
        // 1. Save to the install directory of lidar_solution_bringup package
        try {
            // Use ament_index to find the installed package path
            std::string bringup_package_name = "lidar_solution_bringup";
            std::string bringup_package_path;
            
            try {
                bringup_package_path = ament_index_cpp::get_package_share_directory(bringup_package_name);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to get bringup package directory: %s", e.what());
                return;
            }
            
            std::string config_dir = bringup_package_path + "/config";
            std::string params_file_path = config_dir + "/parameters.yaml";
            
            // Create directory if it doesn't exist
            std::filesystem::create_directories(config_dir);
            
            // First, read the existing file into a string
            std::string yaml_content;
            if (std::filesystem::exists(params_file_path)) {
                std::ifstream file(params_file_path);
                if (file.is_open()) {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    yaml_content = buffer.str();
                    file.close();
                    RCLCPP_INFO(this->get_logger(), "Loaded existing parameters file: %s", params_file_path.c_str());
                }
            }
            
            // If file doesn't exist or couldn't be read, create a basic structure
            if (yaml_content.empty()) {
                yaml_content = "#!check\n\n/**:\n  ros__parameters:\n    global_frame: world\n    lidar_frame: innoviz\n";
            }
            
            // Create our node's parameters section
            std::stringstream node_params;
            node_params << "lidar_tf_position_publisher_node:\n";
            node_params << "  ros__parameters:\n";
            node_params << "    global_frame: " << global_frame_ << "\n";
            node_params << "    lidar_frame: " << lidar_frame_ << "\n";
            node_params << "    lidar_pitch: " << format_double(lidar_pitch_) << "\n";
            node_params << "    lidar_roll: " << format_double(lidar_roll_) << "\n";
            node_params << "    lidar_yaw: " << format_double(lidar_yaw_) << "\n";
            node_params << "    lidar_x: " << format_double(lidar_x_) << "\n";
            node_params << "    lidar_y: " << format_double(lidar_y_) << "\n";
            node_params << "    lidar_z: " << format_double(lidar_z_) << "\n";
            node_params << "    tf_hz: " << format_double(tf_hz_) << "\n";
            
            // Process the existing YAML content line by line
            std::vector<std::string> lines;
            std::istringstream iss(yaml_content);
            std::string line;
            
            // Read all lines into a vector
            while (std::getline(iss, line)) {
                lines.push_back(line);
            }
            
            // Construct the new YAML content
            std::stringstream new_yaml;
            bool in_lidar_tf_section = false;
            bool in_ros_parameters_section = false;
            bool lidar_tf_section_found = false;
            
            for (size_t i = 0; i < lines.size(); ++i) {
                // Check if we're entering the lidar_tf_position_publisher_node section
                if (lines[i] == "lidar_tf_position_publisher_node:" || 
                    lines[i].find("lidar_tf_position_publisher_node:") == 0) {
                    in_lidar_tf_section = true;
                    lidar_tf_section_found = true;
                    new_yaml << "lidar_tf_position_publisher_node:" << "\n";
                    continue;
                }
                
                // Check if we're entering the ros__parameters section within lidar_tf
                if (in_lidar_tf_section && (lines[i] == "  ros__parameters:" || 
                                           lines[i].find("  ros__parameters:") == 0)) {
                    in_ros_parameters_section = true;
                    new_yaml << "  ros__parameters:" << "\n";
                    
                    // Add all our parameters
                    new_yaml << "    global_frame: " << global_frame_ << "\n";
                    new_yaml << "    lidar_frame: " << lidar_frame_ << "\n";
                    new_yaml << "    lidar_pitch: " << format_double(lidar_pitch_) << "\n";
                    new_yaml << "    lidar_roll: " << format_double(lidar_roll_) << "\n";
                    new_yaml << "    lidar_yaw: " << format_double(lidar_yaw_) << "\n";
                    new_yaml << "    lidar_x: " << format_double(lidar_x_) << "\n";
                    new_yaml << "    lidar_y: " << format_double(lidar_y_) << "\n";
                    new_yaml << "    lidar_z: " << format_double(lidar_z_) << "\n";
                    new_yaml << "    tf_hz: " << format_double(tf_hz_) << "\n";
                    
                    // Skip all existing parameter lines
                    while (i + 1 < lines.size() && 
                           !lines[i + 1].empty() && 
                           lines[i + 1].find("    ") == 0) {
                        i++;
                    }
                    continue;
                }
                
                // If we're in the lidar_tf section but not in ros__parameters, check if we're exiting
                if (in_lidar_tf_section) {
                    if (!lines[i].empty() && lines[i][0] != ' ') {
                        // We've reached another top-level section
                        in_lidar_tf_section = false;
                        in_ros_parameters_section = false;
                    } else if (in_ros_parameters_section && 
                              (!lines[i].empty() && lines[i].find("    ") != 0)) {
                        // We've exited the parameters section but still in the node section
                        in_ros_parameters_section = false;
                    }
                }
                
                // Add the line to the output
                new_yaml << lines[i] << "\n";
            }
            
            // If we didn't find the lidar_tf section, add it at the end
            if (!lidar_tf_section_found) {
                new_yaml << "\n" << node_params.str();
            }
            
            // Open file for writing
            std::ofstream bringup_fout(params_file_path);
            if (!bringup_fout.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open bringup parameters file for writing: %s", params_file_path.c_str());
                return;
            }
            
            // Write YAML to file
            bringup_fout << new_yaml.str();
            bringup_fout.close();
            
            RCLCPP_INFO(this->get_logger(), "\033[1;33mParameters saved to bringup package: %s\033[0m", params_file_path.c_str());
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error saving parameters to bringup package: %s", e.what());
        }
        
        // 2. Save to the install directory of lidar_tf_position_publisher package
        try {
            // Use ament_index to find the installed package path
            std::string lidar_tf_package_name = "lidar_tf_position_publisher";
            std::string lidar_tf_package_path;
            
            try {
                lidar_tf_package_path = ament_index_cpp::get_package_share_directory(lidar_tf_package_name);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to get lidar_tf package directory: %s", e.what());
                return;
            }
            
            std::string params_dir = lidar_tf_package_path + "/params";
            std::string params_file_path = params_dir + "/parameters.yaml";
            
            // Create directory if it doesn't exist
            std::filesystem::create_directories(params_dir);
            
            // Helper function to format doubles with decimal point
            auto format_double = [](double value) -> std::string {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(1) << value;
                return oss.str();
            };
            
            // Create our node's parameters section
            std::stringstream node_params;
            node_params << "lidar_tf_position_publisher_node:\n";
            node_params << "  ros__parameters:\n";
            node_params << "    global_frame: " << global_frame_ << "\n";
            node_params << "    lidar_frame: " << lidar_frame_ << "\n";
            node_params << "    lidar_pitch: " << format_double(lidar_pitch_) << "\n";
            node_params << "    lidar_roll: " << format_double(lidar_roll_) << "\n";
            node_params << "    lidar_yaw: " << format_double(lidar_yaw_) << "\n";
            node_params << "    lidar_x: " << format_double(lidar_x_) << "\n";
            node_params << "    lidar_y: " << format_double(lidar_y_) << "\n";
            node_params << "    lidar_z: " << format_double(lidar_z_) << "\n";
            node_params << "    tf_hz: " << format_double(tf_hz_) << "\n";
            
            // For the lidar_tf package, we'll just completely replace the file
            // since it should only contain parameters for this node
            std::ofstream lidar_tf_fout(params_file_path);
            if (!lidar_tf_fout.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open lidar_tf parameters file for writing: %s", params_file_path.c_str());
                return;
            }
            
            // Write YAML to file
            lidar_tf_fout << node_params.str();
            lidar_tf_fout.close();
            
            RCLCPP_INFO(this->get_logger(), "\033[1;33mParameters also saved to lidar_tf package: %s\033[0m", params_file_path.c_str());
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error saving parameters to lidar_tf package: %s", e.what());
        }
    }
    catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Error saving parameters: %s", e.what());
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarTFPublisher>());
    rclcpp::shutdown();
    return 0;
}