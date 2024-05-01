// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_message:msg/FullState.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__STRUCT_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/FullState in the package custom_message.
typedef struct custom_message__msg__FullState
{
  double x;
  double y;
  double yaw;
  double v;
  double omega;
  double delta;
  double throttle;
} custom_message__msg__FullState;

// Struct for a sequence of custom_message__msg__FullState.
typedef struct custom_message__msg__FullState__Sequence
{
  custom_message__msg__FullState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_message__msg__FullState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__FULL_STATE__STRUCT_H_
