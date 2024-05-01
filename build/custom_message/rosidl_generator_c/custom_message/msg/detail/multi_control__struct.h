// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_message:msg/MultiControl.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__STRUCT_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'multi_control'
#include "custom_message/msg/detail/control_inputs__struct.h"

/// Struct defined in msg/MultiControl in the package custom_message.
typedef struct custom_message__msg__MultiControl
{
  custom_message__msg__ControlInputs__Sequence multi_control;
} custom_message__msg__MultiControl;

// Struct for a sequence of custom_message__msg__MultiControl.
typedef struct custom_message__msg__MultiControl__Sequence
{
  custom_message__msg__MultiControl * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_message__msg__MultiControl__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__MULTI_CONTROL__STRUCT_H_
