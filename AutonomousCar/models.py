import numpy as np
from collections.abc import Iterable
from utilFuncs import WeightInitializer, LeakyRelu, Sigmoid
from configs import Constants

class Brain:
    def __init__(
            self,
            layer_sizes: Iterable[int],
            input_size: int,
            output_size: int,
            initialize_weights = True
        ) -> None:
        self.layer_sizes: Iterable[int] = layer_sizes
        self.output_size: int = output_size
        self.input_size: int = input_size

        self.weights = []
        self.biases = []

        if initialize_weights:
            for i in range(len(layer_sizes)):
                if i == 0:
                    previous_layer_size = self.input_size
                else:
                    previous_layer_size = self.layer_sizes[i - 1]
                
                self.weights.append(WeightInitializer((previous_layer_size, self.layer_sizes[i])))
                self.biases.append(WeightInitializer(self.layer_sizes[i]))
            
            self.weights.append(WeightInitializer((self.layer_sizes[-1], self.output_size)))
            self.biases.append(WeightInitializer(self.output_size))


    def predict(
        self,
        inputs: Iterable[float],
    ):
        inputs = np.array(inputs).reshape((-1, 1))

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            inputs = ((inputs * w).sum(axis = 0) + b).reshape((-1, 1))

            if i != len(self.weights) - 1:
                inputs = LeakyRelu(inputs)

        return Sigmoid(inputs.reshape(-1)) - 0.5

class Car:
    def __init__(
        self,
        brain: Brain
    ):
        self.brain: Brain = brain
        self.position: list[float, float] = [0, Constants.CAR_RADIUS]
        self.velocity: list[float, float] = [0, 0]

    def move(self):
        inputs = [
            self.velocity[0] / Constants.MAX_FW_VELOCITY,
            self.velocity[1] / Constants.MAX_LR_VELOCITY,
            Constants.WALL_DISTANCE - self.position[0],
            abs(-Constants.WALL_DISTANCE - self.position[0])
        ]

        preds = self.brain.predict(inputs)

        self.velocity[0] = np.clip(
            self.velocity[0] + (preds[0] * Constants.MAX_FW_ACC),
            0,
            Constants.MAX_FW_VELOCITY
        )

        self.velocity[1] = np.clip(
            self.velocity[1] + (preds[1] * Constants.MAX_LR_ACC),
            -Constants.MAX_LR_VELOCITY,
            Constants.MAX_LR_VELOCITY
        )

        self.position[0] = np.clip(
            self.position[0] + self.velocity[0],
            - Constants.WALL_DISTANCE + Constants.CAR_RADIUS,
            Constants.WALL_DISTANCE - Constants.CAR_RADIUS
        )
        self.position[1] += self.velocity[1]

        return self

    def reset(self):
        self.position = [0, 0]
        self.velocity = [0, 0]
    
    def __repr__(self) -> str:
        return f"Car(position = {self.position}, velocity = {self.velocity})"

class Obstacle:
    def __init__(
            self,
            radius: float,
            position: list[float, float]
    ) -> None:
        self.radius: float = radius
        self.position: list[float, float] = position
