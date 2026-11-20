"""NumPy optimizers for training the MLP."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

ParameterMap: TypeAlias = MutableMapping[str, NDArray[np.floating]]
GradientMap: TypeAlias = Mapping[str, ArrayLike]

__all__ = ["SGD"]


class SGD:
    """Stochastic gradient descent with optional momentum.

    Parameters are updated in place using:

        velocity = momentum * velocity + gradient
        parameter -= learning_rate * velocity

    With ``momentum=0``, this is ordinary SGD.
    """

    def __init__(
        self,
        learning_rate: float,
        *,
        momentum: float = 0.0,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be finite and in the range [0, 1).")

        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self._velocity: dict[str, NDArray[np.floating]] = {}

    def step(
        self,
        parameters: ParameterMap,
        gradients: GradientMap,
    ) -> None:
        """Apply one update to all parameters in place.

        Args:
            parameters: Named, mutable floating-point parameter arrays.
            gradients: Gradient arrays with exactly the same names and shapes.

        Raises:
            ValueError: If parameter/gradient names or shapes do not match.
            TypeError: If parameters or gradients are not real numeric arrays.
            FloatingPointError: If gradients or proposed updates are non-finite.
        """
        if not parameters:
            raise ValueError("parameters must not be empty.")

        parameter_names = set(parameters)
        gradient_names = set(gradients)
        if parameter_names != gradient_names:
            missing = sorted(parameter_names - gradient_names)
            unexpected = sorted(gradient_names - parameter_names)
            raise ValueError(
                f"Gradient keys do not match parameter keys; "
                f"missing={missing}, unexpected={unexpected}."
            )

        if self._velocity:
            if set(self._velocity) != parameter_names:
                raise ValueError(
                    "Parameter names changed after momentum state was initialized."
                )

        proposed_parameters: dict[str, NDArray[np.floating]] = {}
        proposed_velocity: dict[str, NDArray[np.floating]] = {}

        # Validate and calculate every update before mutating any parameter.
        for name, parameter in parameters.items():
            if not isinstance(parameter, np.ndarray):
                raise TypeError(f"Parameter {name!r} must be a NumPy array.")
            if not np.issubdtype(parameter.dtype, np.floating):
                raise TypeError(
                    f"Parameter {name!r} must have a floating dtype; "
                    f"got {parameter.dtype}."
                )
            if not parameter.flags.writeable:
                raise ValueError(f"Parameter {name!r} is not writable.")

            gradient = np.asarray(gradients[name])
            if not np.issubdtype(gradient.dtype, np.number):
                raise TypeError(f"Gradient {name!r} must be numeric.")
            if np.issubdtype(gradient.dtype, np.complexfloating):
                raise TypeError(f"Gradient {name!r} must not be complex.")
            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"Gradient {name!r} has shape {gradient.shape}; "
                    f"expected {parameter.shape}."
                )
            if not np.isfinite(gradient).all():
                raise FloatingPointError(
                    f"Gradient {name!r} contains non-finite values."
                )

            gradient = gradient.astype(parameter.dtype, copy=False)
            if not np.isfinite(gradient).all():
                raise FloatingPointError(
                    f"Gradient {name!r} is non-finite in dtype {parameter.dtype}."
                )

            if self.momentum:
                if name in self._velocity:
                    velocity = self._velocity[name]
                    if velocity.shape != parameter.shape:
                        raise ValueError(
                            f"Momentum state for {name!r} has shape "
                            f"{velocity.shape}; expected {parameter.shape}."
                        )
                    velocity = (
                        self.momentum * velocity + gradient
                    ).astype(parameter.dtype, copy=False)
                else:
                    velocity = gradient.copy()

                update = self.learning_rate * velocity
                proposed_velocity[name] = velocity
            else:
                update = self.learning_rate * gradient

            updated_parameter = parameter - update
            if not np.isfinite(updated_parameter).all():
                raise FloatingPointError(
                    f"Update for {name!r} would produce non-finite parameters."
                )

            proposed_parameters[name] = updated_parameter

        for name, parameter in parameters.items():
            np.copyto(parameter, proposed_parameters[name], casting="no")

        if self.momentum:
            self._velocity = proposed_velocity

    def reset_state(self) -> None:
        """Clear accumulated momentum, for example before a fresh training run."""
        self._velocity.clear()