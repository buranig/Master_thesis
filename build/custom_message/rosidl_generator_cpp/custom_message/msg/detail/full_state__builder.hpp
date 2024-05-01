// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_message:msg/FullState.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__BUILDER_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_message/msg/detail/full_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_message
{

namespace msg
{

namespace builder
{

class Init_FullState_throttle
{
public:
  explicit Init_FullState_throttle(::custom_message::msg::FullState & msg)
  : msg_(msg)
  {}
  ::custom_message::msg::FullState throttle(::custom_message::msg::FullState::_throttle_type arg)
  {
    msg_.throttle = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_message::msg::FullState msg_;
};

class Init_FullState_delta
{
public:
  explicit Init_FullState_delta(::custom_message::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_throttle delta(::custom_message::msg::FullState::_delta_type arg)
  {
    msg_.delta = std::move(arg);
    return Init_FullState_throttle(msg_);
  }

private:
  ::custom_message::msg::FullState msg_;
};

class Init_FullState_omega
{
public:
  explicit Init_FullState_omega(::custom_message::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_delta omega(::custom_message::msg::FullState::_omega_type arg)
  {
    msg_.omega = std::move(arg);
    return Init_FullState_delta(msg_);
  }

private:
  ::custom_message::msg::FullState msg_;
};

class Init_FullState_v
{
public:
  explicit Init_FullState_v(::custom_message::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_omega v(::custom_message::msg::FullState::_v_type arg)
  {
    msg_.v = std::move(arg);
    return Init_FullState_omega(msg_);
  }

private:
  ::custom_message::msg::FullState msg_;
};

class Init_FullState_yaw
{
public:
  explicit Init_FullState_yaw(::custom_message::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_v yaw(::custom_message::msg::FullState::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_FullState_v(msg_);
  }

private:
  ::custom_message::msg::FullState msg_;
};

class Init_FullState_y
{
public:
  explicit Init_FullState_y(::custom_message::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_yaw y(::custom_message::msg::FullState::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_FullState_yaw(msg_);
  }

private:
  ::custom_message::msg::FullState msg_;
};

class Init_FullState_x
{
public:
  Init_FullState_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FullState_y x(::custom_message::msg::FullState::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_FullState_y(msg_);
  }

private:
  ::custom_message::msg::FullState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_message::msg::FullState>()
{
  return custom_message::msg::builder::Init_FullState_x();
}

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__BUILDER_HPP_
