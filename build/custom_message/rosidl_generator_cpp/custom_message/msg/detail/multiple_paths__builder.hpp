// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_message:msg/MultiplePaths.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__BUILDER_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_message/msg/detail/multiple_paths__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_message
{

namespace msg
{

namespace builder
{

class Init_MultiplePaths_multiple_path
{
public:
  Init_MultiplePaths_multiple_path()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::custom_message::msg::MultiplePaths multiple_path(::custom_message::msg::MultiplePaths::_multiple_path_type arg)
  {
    msg_.multiple_path = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_message::msg::MultiplePaths msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_message::msg::MultiplePaths>()
{
  return custom_message::msg::builder::Init_MultiplePaths_multiple_path();
}

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__BUILDER_HPP_
