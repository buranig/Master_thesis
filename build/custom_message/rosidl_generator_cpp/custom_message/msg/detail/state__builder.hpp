// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_message:msg/State.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__STATE__BUILDER_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_message/msg/detail/state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_message
{

namespace msg
{

namespace builder
{

class Init_State_omega
{
public:
  explicit Init_State_omega(::custom_message::msg::State & msg)
  : msg_(msg)
  {}
  ::custom_message::msg::State omega(::custom_message::msg::State::_omega_type arg)
  {
    msg_.omega = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_message::msg::State msg_;
};

class Init_State_v
{
public:
  explicit Init_State_v(::custom_message::msg::State & msg)
  : msg_(msg)
  {}
  Init_State_omega v(::custom_message::msg::State::_v_type arg)
  {
    msg_.v = std::move(arg);
    return Init_State_omega(msg_);
  }

private:
  ::custom_message::msg::State msg_;
};

class Init_State_yaw
{
public:
  explicit Init_State_yaw(::custom_message::msg::State & msg)
  : msg_(msg)
  {}
  Init_State_v yaw(::custom_message::msg::State::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_State_v(msg_);
  }

private:
  ::custom_message::msg::State msg_;
};

class Init_State_y
{
public:
  explicit Init_State_y(::custom_message::msg::State & msg)
  : msg_(msg)
  {}
  Init_State_yaw y(::custom_message::msg::State::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_State_yaw(msg_);
  }

private:
  ::custom_message::msg::State msg_;
};

class Init_State_x
{
public:
  Init_State_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_State_y x(::custom_message::msg::State::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_State_y(msg_);
  }

private:
  ::custom_message::msg::State msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_message::msg::State>()
{
  return custom_message::msg::builder::Init_State_x();
}

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__STATE__BUILDER_HPP_
