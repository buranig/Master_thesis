#include "rclcpp/rclcpp.hpp"
#include "lar_msgs/srv/car_predict.hpp"
#include "model_driving/DrivingDynamicsEigen.hpp"

// using namespace ;
using Model = driving_model::eigen_model::KinematicBicycleWithActuators;











void handle_vehicle_state(
        const std::shared_ptr<lar_msgs::srv::CarPredict::Request> request,
        std::shared_ptr<lar_msgs::srv::CarPredict::Response> response)
    {
        std::cout << "Received request" << std::endl;
        // Model::State x0({request->x, request->y, request->theta, request->v, request->omega});
        // Model::Input u({request->u1, request->u2});

        // Model::State x_dot;
        // Model::Param p = {};
        // Model::DynOut dyn_out;
        // x_dot = model.dynamics<double>(x0, u, p, dyn_out);

        // // Update the state with the time step
        // Model::State x1 = x0 + request->deltaT * x_dot;

        // response->x = x1[0];
        // response->y = x1[1];
        // response->theta = x1[2];
        // response->v = x1[3];
        // response->omega = x1[4];
    }



class VehicleStateService : public rclcpp::Node
{
public:
    VehicleStateService(): Node("vehicle_predict_state")
    {
        std::string car_path_;
        // Get the car_path parameter
        this->declare_parameter<std::string>("car_path", "");
        this->get_parameter("car_path", car_path_);

        model.load_params_from_yaml(car_path_);

        service_ = 
             this->create_service<lar_msgs::srv::CarPredict>("add_three_ints",  &handle_vehicle_state);
    }

    

    rclcpp::Service<lar_msgs::srv::CarPredict>::SharedPtr service_;
    Model model;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<rclcpp::Node> node = std::make_shared<VehicleStateService>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}