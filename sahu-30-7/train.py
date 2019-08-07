import torch, torchvision
from data import ImageDataset
from model import shufflenet
from torch.utils.data import DataLoader
from torchsummary import summary
import time
from tensorboard_logger import configure, log_value

configure("logs\shufflenet-v1")

train_path = "D:\\imp_data\\labels\\train.txt"
test_path = "D:\\imp_data\\labels\\test.txt"
val_path = "D:\\imp_data\\labels\\val.txt"

train_dataset = ImageDataset(train_path)
test_dataset = ImageDataset(test_path)
val_dataset = ImageDataset(val_path)

bs = 12 #batch size
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True)
test_data_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, pin_memory=True)
val_data_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True, pin_memory=True)

def train_and_validate(model, loss_criterion, optimizer, epochs):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_acc = 0.0

    

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            log_value('Train loss', loss.item(), i)
            log_value('Train Accuracy', acc.item(), i)


            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(val_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        log_value('avg_train_loss', avg_train_loss, epoch)
        log_value('avg_train_acc',avg_train_acc, epoch)
        log_value('avg_valid_loss', avg_valid_loss, epoch)
        log_value('avg_valid_acc', avg_valid_acc, epoch)

            
        # Save if the model has best accuracy till now
        torch.save(model,'models/30-7/'+dataset+'_model_'+str(epoch)+'.pt')
            
    return model, history

# Define Optimizer and Loss Function
loss_func = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(shufflenet.parameters())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print the model to be trained
summary(shufflenet, input_size=(3, 1000, 1000), batch_size=bs, device='cuda')

# Train the model for 25 epochs
num_epochs = 2
trained_model, history = train_and_validate(shufflenet, loss_func, optimizer, num_epochs)

torch.save(history, dataset+'_history.pt')    