// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_message:msg/MultiState.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__BUILDER_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_message/msg/detail/multi_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_message
{

namespace msg
{

namespace builder
{

class Init_MultiState_multiple_state
{
public:
  Init_MultiState_multiple_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::custom_message::msg::MultiState multiple_state(::custom_message::msg::MultiState::_multiple_state_type arg)
  {
    msg_.multiple_state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_message::msg::MultiState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_message::msg::MultiState>()
{
  return custom_message::msg::builder::Init_MultiState_multiple_state();
}

}  // namespace custom_message

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTI_STATE__BUILDER_HPP_
