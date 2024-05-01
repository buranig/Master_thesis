// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_message:msg/Coordinate.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__BUILDER_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_message/msg/detail/coordinate__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_message
{

namespace msg
{

namespace builder
{

class Init_Coordinate_y
{
public:
  explicit Init_Coordinate_y(::custom_message::msg::Coordinate & msg)
  : msg_(msg)
  {}
  ::custom_message::msg::Coordinate y(::custom_message::msg::Coordinate::_y_type arg)
  {
    msg_.y = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_message::msg::Coordinate msg_;
};

class Init_Coordinate_x
{
public:
  Init_Coordinate_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Coordinate_y x(::custom_message::msg::Coordinate::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_Coordinate_y(msg_);
  }

private:
  ::custom_message::msg::Coordinate msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_message::msg::Coordinate>()
{
  return custom_message::msg::builder::Init_Coordinate_x();
}

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__COORDINATE__BUILDER_HPP_
