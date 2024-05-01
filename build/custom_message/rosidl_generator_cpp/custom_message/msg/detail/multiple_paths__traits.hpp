// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_message:msg/MultiplePaths.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__TRAITS_HPP_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_message/msg/detail/multiple_paths__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'multiple_path'
#include "custom_message/msg/detail/path__traits.hpp"

namespace custom_message
{

namespace msg
{

inline void to_flow_style_yaml(
  const MultiplePaths & msg,
  std::ostream & out)
{
  out << "{";
  // member: multiple_path
  {
    if (msg.multiple_path.size() == 0) {
      out << "multiple_path: []";
    } else {
      out << "multiple_path: [";
      size_t pending_items = msg.multiple_path.size();
      for (auto item : msg.multiple_path) {
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
  const MultiplePaths & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: multiple_path
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.multiple_path.size() == 0) {
      out << "multiple_path: []\n";
    } else {
      out << "multiple_path:\n";
      for (auto item : msg.multiple_path) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MultiplePaths & msg, bool use_flow_style = false)
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
  const custom_message::msg::MultiplePaths & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_message::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_message::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_message::msg::MultiplePaths & msg)
{
  return custom_message::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_message::msg::MultiplePaths>()
{
  return "custom_message::msg::MultiplePaths";
}

template<>
inline const char * name<custom_message::msg::MultiplePaths>()
{
  return "custom_message/msg/MultiplePaths";
}

template<>
struct has_fixed_size<custom_message::msg::MultiplePaths>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<custom_message::msg::MultiplePaths>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<custom_message::msg::MultiplePaths>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTIPLE_PATHS__TRAITS_HPP_
