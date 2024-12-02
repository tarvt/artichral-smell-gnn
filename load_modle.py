# To load the model later
model = GNN(num_features=dataset.num_node_features, num_classes=dataset.num_classes)
model.load_state_dict(torch.load('path_to_save_model/gnn_model.pth'))
model.eval()  # Set the model to evaluation mode