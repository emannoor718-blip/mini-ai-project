from abc import ABC, abstractmethod

class ModelConfig:
    def __init__(self, model_name, learning_rate=0.01, epochs=10):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def __repr__(self):
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"


class BaseModel(ABC):
    model_count = 0
    
    def __init__(self, config: ModelConfig):
        self.config = config
        BaseModel.model_count += 1
    
    @abstractmethod
    def train(self, data):
        pass
    
    @abstractmethod
    def evaluate(self, data):
        pass


class LinearRegressionModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def train(self, data):
        samples = len(data)
        print(f"LinearRegression: Training on {samples} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")
    
    def evaluate(self, data):
        print(f"LinearRegression: Evaluation MSE = 0.042")


class NeuralNetworkModel(BaseModel):
    def __init__(self, config: ModelConfig, layers: list):
        super().__init__(config)
        self.layers = layers
    
    def train(self, data):
        samples = len(data)
        print(f"NeuralNetwork {self.layers}: Training on {samples} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")
    
    def evaluate(self, data):
        print(f"NeuralNetwork: Evaluation Accuracy = 91.5%")


class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def get_data(self):
        return self.dataset


class Trainer:
    def __init__(self, model: BaseModel, data_loader: DataLoader):
        self.model = model
        self.data_loader = data_loader
    
    def run(self):
        print(f"\n--- Training {self.model.config.model_name} ---")
        data = self.data_loader.get_data()
        self.model.train(data)
        self.model.evaluate(data)


# Main execution
if __name__ == "__main__":
    lr_config = ModelConfig("LinearRegression", learning_rate=0.01, epochs=10)
    nn_config = ModelConfig("NeuralNetwork", learning_rate=0.001, epochs=20)
    
    print(lr_config)
    print(nn_config)
    
    lr_model = LinearRegressionModel(lr_config)
    nn_model = NeuralNetworkModel(nn_config, layers=[64, 32, 1])
    
    print(f"Models created: {BaseModel.model_count}")
    
    dataset = [1, 2, 3, 4, 5]
    data_loader = DataLoader(dataset)
    
    trainer1 = Trainer(lr_model, data_loader)
    trainer1.run()
    
    trainer2 = Trainer(nn_model, data_loader)
    trainer2.run()