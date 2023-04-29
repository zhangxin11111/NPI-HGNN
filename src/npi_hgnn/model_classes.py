from torch_geometric.nn import TopKPooling, SAGEConv,RGCNConv,RGATConv,SAGPooling
import torch
from torch_geometric.nn import global_max_pool,global_mean_pool,Set2Set,global_sort_pool,BatchNorm
import torch.nn.functional as F

class Model_1(torch.nn.Module):
    def __init__(self, num_node_features, num_relations,num_bases, num_of_classes=2):
        super(Model_1, self).__init__()
        self.batchNorm = BatchNorm(num_node_features)
        # RGCN
        self.conv1_1 = RGCNConv(in_channels=num_node_features,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool1_1 = TopKPooling(128, ratio=0.5)
        self.conv2_1 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool2_1 = TopKPooling(128, ratio=0.5)
        self.conv3_1 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool3_1 = TopKPooling(128, ratio=0.5)
        # RGCN
        self.conv1_2 = RGCNConv(in_channels=num_node_features,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool1_2 = TopKPooling(128, ratio=0.5)
        self.conv2_2 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool2_2 = TopKPooling(128, ratio=0.5)
        self.conv3_2 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool3_2 = TopKPooling(128, ratio=0.5)
        self.global_2 = Set2Set(128, processing_steps=3, num_layers=1)
        # RGCN
        self.conv1_3 = RGCNConv(in_channels=num_node_features,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool1_3 = TopKPooling(128, ratio=0.5)
        self.conv2_3 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool2_3 = TopKPooling(128, ratio=0.5)
        self.conv3_3 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool3_3 = TopKPooling(128, ratio=0.5)
        self.global_3 = global_sort_pool

        self.lin1 = torch.nn.Linear(128*5, 128)

        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_of_classes)

    def forward(self, data):
        x, edge_index , batch ,edge_type_list = data.x, data.edge_index, data.batch.to(dtype=torch.int64),data.edge_attr
        x = self.batchNorm(x)
        # RGCN
        x_t = F.leaky_relu(self.conv1_1(x,edge_index,edge_type_list))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool1_1(x_t, edge_index, edge_attr=edge_type_list, batch=batch)
        res1_1= torch.cat([global_max_pool(x_t,batch_t), global_mean_pool(x_t,batch_t)], dim=1)
        x_t = F.leaky_relu(self.conv2_1(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool2_1(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res2_1 = torch.cat([global_max_pool(x_t,batch_t), global_mean_pool(x_t,batch_t)], dim=1)
        x_t = F.leaky_relu(self.conv3_1(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool3_1(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res3_1 = torch.cat([global_max_pool(x_t,batch_t), global_mean_pool(x_t,batch_t)], dim=1)
        # RGCN
        x_t = F.leaky_relu(self.conv1_2(x,edge_index,edge_type_list))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool1_2(x_t, edge_index, edge_attr=edge_type_list, batch=batch)
        res1_2= self.global_2(x_t,batch_t)
        x_t = F.leaky_relu(self.conv2_2(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool2_2(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res2_2 = self.global_2(x_t,batch_t)
        x_t = F.leaky_relu(self.conv3_2(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool3_2(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res3_2 = self.global_2(x_t,batch_t)
        # RGCN
        x_t = F.leaky_relu(self.conv1_3(x,edge_index,edge_type_list))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool1_3(x_t, edge_index, edge_attr=edge_type_list, batch=batch)
        res1_3= self.global_3(x_t,batch_t,1)
        x_t = F.leaky_relu(self.conv2_3(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool2_3(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res2_3 = self.global_3(x_t,batch_t,1)
        x_t = F.leaky_relu(self.conv3_3(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool3_3(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res3_3 = self.global_3(x_t,batch_t,1)
        x= torch.cat([res1_1+res2_1+res3_1,res1_2+res2_2+res3_2,res1_3+res2_3+res3_3],dim=1) #加入残差
        x = F.leaky_relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

class Model_2(torch.nn.Module):
    def __init__(self, num_node_features, num_relations,num_bases, num_of_classes=2):
        super(Model_2, self).__init__()
        self.batchNorm = BatchNorm(num_node_features)
        # RGCN
        self.conv1_1 = RGCNConv(in_channels=num_node_features,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool1_1 = TopKPooling(128, ratio=0.5)
        self.conv2_1 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool2_1 = TopKPooling(128, ratio=0.5)
        self.conv3_1 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool3_1 = TopKPooling(128, ratio=0.5)
        self.global_1 = global_max_pool
        # RGCN
        self.conv1_2 = RGCNConv(in_channels=num_node_features,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool1_2 = TopKPooling(128, ratio=0.5)
        self.conv2_2 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool2_2 = TopKPooling(128, ratio=0.5)
        self.conv3_2 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool3_2 = TopKPooling(128, ratio=0.5)
        self.global_2 = global_mean_pool
        # RGCN
        self.conv1_3 = RGCNConv(in_channels=num_node_features,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool1_3 = TopKPooling(128, ratio=0.5)
        self.conv2_3 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool2_3 = TopKPooling(128, ratio=0.5)
        self.conv3_3 = RGCNConv(in_channels=128,out_channels=128,num_relations=num_relations,num_bases=num_bases)
        self.pool3_3 = TopKPooling(128, ratio=0.5)
        self.global_3 = global_sort_pool

        self.lin1 = torch.nn.Linear(128*3, 128)

        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_of_classes)

    def forward(self, data):
        x, edge_index , batch ,edge_type_list = data.x, data.edge_index, data.batch.to(dtype=torch.int64),data.edge_attr
        x=self.batchNorm(x)
        # RGCN
        x_t = F.relu(self.conv1_1(x,edge_index,edge_type_list))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool1_1(x_t, edge_index, edge_attr=edge_type_list, batch=batch)
        res1_1= self.global_1(x_t,batch_t)
        x_t = F.relu(self.conv2_1(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool2_1(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res2_1 = self.global_1(x_t,batch_t)
        x_t = F.relu(self.conv3_1(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool3_1(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res3_1 = self.global_1(x_t,batch_t)
        # RGCN
        x_t = F.relu(self.conv1_2(x,edge_index,edge_type_list))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool1_2(x_t, edge_index, edge_attr=edge_type_list, batch=batch)
        res1_2= self.global_2(x_t,batch_t)
        x_t = F.relu(self.conv2_2(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool2_2(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res2_2 = self.global_2(x_t,batch_t)
        x_t = F.relu(self.conv3_2(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool3_2(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res3_2 = self.global_2(x_t,batch_t)
        # RGCN
        x_t = F.relu(self.conv1_3(x,edge_index,edge_type_list))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool1_3(x_t, edge_index, edge_attr=edge_type_list, batch=batch)
        res1_3= self.global_3(x_t,batch_t,1)
        x_t = F.relu(self.conv2_3(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool2_3(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res2_3 = self.global_3(x_t,batch_t,1)
        x_t = F.relu(self.conv3_3(x_t,edge_index_t,edge_type_list_t))
        x_t, edge_index_t, edge_type_list_t, batch_t, _, _ = self.pool3_3(x_t, edge_index_t, edge_attr=edge_type_list_t, batch=batch_t)
        res3_3 = self.global_3(x_t,batch_t,1)
        x= torch.cat([res1_1+res2_1+res3_1,res1_2+res2_2+res3_2,res1_3+res2_3+res3_3],dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x