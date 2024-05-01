// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from custom_message:msg/ControlInputs.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__FUNCTIONS_H_
#define CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "custom_message/msg/rosidl_generator_c__visibility_control.h"

#include "custom_message/msg/detail/control_inputs__struct.h"

/// Initialize msg/ControlInputs message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * custom_message__msg__ControlInputs
 * )) before or use
 * custom_message__msg__ControlInputs__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__ControlInputs__init(custom_message__msg__ControlInputs * msg);

/// Finalize msg/ControlInputs message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__ControlInputs__fini(custom_message__msg__ControlInputs * msg);

/// Create msg/ControlInputs message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * custom_message__msg__ControlInputs__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
custom_message__msg__ControlInputs *
custom_message__msg__ControlInputs__create();

/// Destroy msg/ControlInputs message.
/**
 * It calls
 * custom_message__msg__ControlInputs__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__ControlInputs__destroy(custom_message__msg__ControlInputs * msg);

/// Check for msg/ControlInputs message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__ControlInputs__are_equal(const custom_message__msg__ControlInputs * lhs, const custom_message__msg__ControlInputs * rhs);

/// Copy a msg/ControlInputs message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__ControlInputs__copy(
  const custom_message__msg__ControlInputs * input,
  custom_message__msg__ControlInputs * output);

/// Initialize array of msg/ControlInputs messages.
/**
 * It allocates the memory for the number of elements and calls
 * custom_message__msg__ControlInputs__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__ControlInputs__Sequence__init(custom_message__msg__ControlInputs__Sequence * array, size_t size);

/// Finalize array of msg/ControlInputs messages.
/**
 * It calls
 * custom_message__msg__ControlInputs__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__ControlInputs__Sequence__fini(custom_message__msg__ControlInputs__Sequence * array);

/// Create array of msg/ControlInputs messages.
/**
 * It allocates the memory for the array and calls
 * custom_message__msg__ControlInputs__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
custom_message__msg__ControlInputs__Sequence *
custom_message__msg__ControlInputs__Sequence__create(size_t size);

/// Destroy array of msg/ControlInputs messages.
/**
 * It calls
 * custom_message__msg__ControlInputs__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
void
custom_message__msg__ControlInputs__Sequence__destroy(custom_message__msg__ControlInputs__Sequence * array);

/// Check for msg/ControlInputs message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__ControlInputs__Sequence__are_equal(const custom_message__msg__ControlInputs__Sequence * lhs, const custom_message__msg__ControlInputs__Sequence * rhs);

/// Copy an array of msg/ControlInputs messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_custom_message
bool
custom_message__msg__ControlInputs__Sequence__copy(
  const custom_message__msg__ControlInputs__Sequence * input,
  custom_message__msg__ControlInputs__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MESSAGE__MSG__DETAIL__CONTROL_INPUTS__FUNCTIONS_H_
