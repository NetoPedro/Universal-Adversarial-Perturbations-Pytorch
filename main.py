import trainer as trainer_module
import data_loader
import matplotlib.pyplot as plt
import adversarial_perturbation
def main():
    trainer = trainer_module.trainer()
    trainset,testset  = data_loader.load_data()
    accuracy = trainer.train(trainset,testset)
    trainset, testset = data_loader.load_data()

   
    v, fooling_rates, accuracies, total_iterations=adversarial_perturbation.generate(accuracy,trainset, testset, trainer.net)

    plt.title("Fooling Rates over Universal Iterations")
    plt.xlabel("Universal Algorithm Iter")
    plt.ylabel("Fooling Rate on test data")
    plt.plot(total_iterations,fooling_rates)
    plt.show()


    plt.title("Accuracy over Universal Iterations")
    plt.xlabel("Universal Algorithm Iter")
    plt.ylabel("Accuracy on Test data")
    plt.plot(total_iterations, accuracies)
    plt.show()



if __name__ == "__main__":
    main()