// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_message:msg/ControlInputs.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__BUILDER_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_message/msg/detail/control_inputs__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_message
{

namespace msg
{

namespace builder
{

class Init_ControlInputs_delta
{
public:
  explicit Init_ControlInputs_delta(::custom_message::msg::ControlInputs & msg)
  : msg_(msg)
  {}
  ::custom_message::msg::ControlInputs delta(::custom_message::msg::ControlInputs::_delta_type arg)
  {
    msg_.delta = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_message::msg::ControlInputs msg_;
};

class Init_ControlInputs_throttle
{
public:
  Init_ControlInputs_throttle()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ControlInputs_delta throttle(::custom_message::msg::ControlInputs::_throttle_type arg)
  {
    msg_.throttle = std::move(arg);
    return Init_ControlInputs_delta(msg_);
  }

private:
  ::custom_message::msg::ControlInputs msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_message::msg::ControlInputs>()
{
  return custom_message::msg::builder::Init_ControlInputs_throttle();
}

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__BUILDER_HPP_
