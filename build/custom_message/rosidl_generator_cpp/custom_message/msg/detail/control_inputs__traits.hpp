// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_message:msg/ControlInputs.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__TRAITS_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_message/msg/detail/control_inputs__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace custom_message
{

namespace msg
{

inline void to_flow_style_yaml(
  const ControlInputs & msg,
  std::ostream & out)
{
  out << "{";
  // member: throttle
  {
    out << "throttle: ";
    rosidl_generator_traits::value_to_yaml(msg.throttle, out);
    out << ", ";
  }

  // member: delta
  {
    out << "delta: ";
    rosidl_generator_traits::value_to_yaml(msg.delta, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ControlInputs & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: throttle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "throttle: ";
    rosidl_generator_traits::value_to_yaml(msg.throttle, out);
    out << "\n";
  }

  // member: delta
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "delta: ";
    rosidl_generator_traits::value_to_yaml(msg.delta, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ControlInputs & msg, bool use_flow_style = false)
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
  const custom_message::msg::ControlInputs & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_message::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_message::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_message::msg::ControlInputs & msg)
{
  return custom_message::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_message::msg::ControlInputs>()
{
  return "custom_message::msg::ControlInputs";
}

template<>
inline const char * name<custom_message::msg::ControlInputs>()
{
  return "custom_message/msg/ControlInputs";
}

template<>
struct has_fixed_size<custom_message::msg::ControlInputs>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<custom_message::msg::ControlInputs>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<custom_message::msg::ControlInputs>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__TRAITS_HPP_
