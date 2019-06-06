import trainer as trainer_module
import data_loader
def main():
    trainer = trainer_module.trainer()
    trainset,testset  = data_loader.load_data()
    trainer.train(trainset,testset)

if __name__ == "__main__":
    main()