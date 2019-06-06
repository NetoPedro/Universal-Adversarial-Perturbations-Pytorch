import trainer as trainer_module
import data_loader
import adversarial_perturbation
def main():
    trainer = trainer_module.trainer()
    trainset,testset  = data_loader.load_data()
    trainer.train(trainset,testset)
    adversarial_perturbation.generate("./fashion_mnist",trainset, testset, trainer.net)

if __name__ == "__main__":
    main()