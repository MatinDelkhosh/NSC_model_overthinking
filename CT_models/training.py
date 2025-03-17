from LTCCNN import *

# -------------------------
#      Training Loop
# -------------------------

if __name__ == "__main__":
    import pickle
    train_data_temp = pickle.load(open('Data/processed_data.pk','rb'))
    train_data = []
    for d in train_data_temp:
        train_data += d

    train_dataset = MazeDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    # Instantiate the network.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MazeSolverNetDynamic(maze_img_size=27, constant_dim=15, cnn_out_dim=16,
                                constant_out_dim=3, ltc_hidden_size=64, ltc_output_dim=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 50
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (maze_seq, labels, constants, time_stamps) in enumerate(train_loader):
            # maze_seq: (batch, T, 2, 27, 27)
            # labels: (batch, T, 3)
            # constants: (batch, 15)
            # time_stamps: (batch, T)
            maze_seq = maze_seq.to(device)
            labels = labels.to(device)
            constants = constants.to(device)

            optimizer.zero_grad()
            outputs, h1, h2 = model(maze_seq, constants, time_stamps)
            print(f'\rforwarded feed {batch_idx}',end=15*' ')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f'\rbackpropd feed {batch_idx} loss: {loss.item():.4f}',end='')
            running_loss += loss.item()
            loss_history.append(loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"\rEpoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}",end=10*' '+'\n')

    print("Training complete.")
    torch.save(model, "CT_models/maze_solver_test3_FixedData.pth")
    print("Saved the model.")

    import matplotlib.pyplot as plt

    # Plot the loss
    plt.plot(loss_history)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Batches")
    plt.show()
