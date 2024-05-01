// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_message:msg/MultiControl.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__BUILDER_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_message/msg/detail/multi_control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_message
{

namespace msg
{

namespace builder
{

class Init_MultiControl_multi_control
{
public:
  Init_MultiControl_multi_control()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::custom_message::msg::MultiControl multi_control(::custom_message::msg::MultiControl::_multi_control_type arg)
  {
    msg_.multi_control = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_message::msg::MultiControl msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_message::msg::MultiControl>()
{
  return custom_message::msg::builder::Init_MultiControl_multi_control();
}

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__BUILDER_HPP_
