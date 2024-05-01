// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_message:msg/MultiControl.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__TRAITS_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_message/msg/detail/multi_control__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'multi_control'
#include "custom_message/msg/detail/control_inputs__traits.hpp"

namespace custom_message
{

namespace msg
{

inline void to_flow_style_yaml(
  const MultiControl & msg,
  std::ostream & out)
{
  out << "{";
  // member: multi_control
  {
    if (msg.multi_control.size() == 0) {
      out << "multi_control: []";
    } else {
      out << "multi_control: [";
      size_t pending_items = msg.multi_control.size();
      for (auto item : msg.multi_control) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MultiControl & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: multi_control
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.multi_control.size() == 0) {
      out << "multi_control: []\n";
    } else {
      out << "multi_control:\n";
      for (auto item : msg.multi_control) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MultiControl & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace custom_message

namespace rosidl_generator_traits
{

[[deprecated("use custom_message::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const custom_message::msg::MultiControl & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_message::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_message::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_message::msg::MultiControl & msg)
{
  return custom_message::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_message::msg::MultiControl>()
{
  return "custom_message::msg::MultiControl";
}

template<>
inline const char * name<custom_message::msg::MultiControl>()
{
  return "custom_message/msg/MultiControl";
}

template<>
struct has_fixed_size<custom_message::msg::MultiControl>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<custom_message::msg::MultiControl>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<custom_message::msg::MultiControl>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__TRAITS_HPP_
