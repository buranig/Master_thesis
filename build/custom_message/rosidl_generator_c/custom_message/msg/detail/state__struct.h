// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_message:msg/State.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__STATE__STRUCT_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/State in the package custom_message.
typedef struct custom_message__msg__State
{
  double x;
  double y;
  double yaw;
  double v;
  double omega;
} custom_message__msg__State;

// Struct for a sequence of custom_message__msg__State.
typedef struct custom_message__msg__State__Sequence
{
  custom_message__msg__State * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_message__msg__State__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__STATE__STRUCT_H_
