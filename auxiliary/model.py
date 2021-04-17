from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from my_utils import sampleSphere
import trimesh
import pointcloud_processor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import copy

import pdb

"""
Template Discovery -> Patch deform
Tamplate learning -> Point translation 
"""


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PointNetfeat(nn.Module):
    def __init__(self, npoint=2500, nlatent=1024):
        """
        Encoder
        input shape: [batch_size, npoints, 3]
        output shape: [batch_size, 1024]
        """

        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        # batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)


class Predictor_unit(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(emb_dim, emb_dim)

        self.fc_weight = nn.Linear(emb_dim, input_dim * output_dim)
        self.fc_scale = nn.Linear(emb_dim, output_dim)
        self.fc_bias = nn.Linear(emb_dim, output_dim)
        self.fc_bias_2 = nn.Linear(emb_dim, output_dim)

        self.bn = nn.BatchNorm1d(emb_dim)
        self.bn_weight = nn.BatchNorm1d(input_dim * output_dim)
        self.bn_scale = nn.BatchNorm1d(output_dim)
        self.bn_bias = nn.BatchNorm1d(output_dim)
        self.bn_bias_2 = nn.BatchNorm1d(output_dim)

    def forward(self, x, layer_name):
        parameters = dict()
        x = F.relu(self.bn(self.fc(x)))

        w = self.bn_weight(self.fc_weight(x))
        parameters[layer_name + '_w'] = w.view(-1, 1, self.output_dim, self.input_dim)

        b = self.bn_bias(self.fc_bias(x))
        parameters[layer_name + '_b'] = b.view(-1, 1, self.output_dim, 1)

        s = self.bn_scale(self.fc_scale(x))
        parameters[layer_name + '_s'] = s.view(-1, 1, self.output_dim, 1)

        b2 = self.bn_bias_2(self.fc_bias_2(x))
        parameters[layer_name + '_b2'] = b2.view(-1, 1, self.output_dim, 1)
        return parameters


class Predictor(nn.Module):
    '''
    Meta-Learning, learn the parameters for decoder
    input: embeddings
    output: decoder's parameters
    '''

    def __init__(self, input_dim=3, output_dim=4, bottleneck=64, emb_dim=1024):
        super().__init__()
        self.layer1 = Predictor_unit(input_dim=3, output_dim=64, emb_dim=1024)
        self.layer2 = Predictor_unit(input_dim=64, output_dim=64, emb_dim=1024)
        self.layer3 = Predictor_unit(input_dim=64, output_dim=64, emb_dim=1024)
        self.layer4 = Predictor_unit(input_dim=64, output_dim=64, emb_dim=1024)
        self.layer5 = Predictor_unit(input_dim=64, output_dim=64, emb_dim=1024)
        self.layer6 = Predictor_unit(input_dim=64, output_dim=4, emb_dim=1024)

    def forward(self, x):
        parameters = dict()
        out = self.layer1(x, 'mlp1')
        parameters = {**parameters, **out}  # merge dict
        out = self.layer2(x, 'mlp2')
        parameters = {**parameters, **out}
        out = self.layer3(x, 'mlp3')
        parameters = {**parameters, **out}
        out = self.layer4(x, 'mlp4')
        parameters = {**parameters, **out}
        out = self.layer5(x, 'mlp5')
        parameters = {**parameters, **out}
        out = self.layer6(x, 'mlp6')
        parameters = {**parameters, **out}

        return parameters


class Meta_Decoder(nn.Module):
    '''
    Decoder;
    input: template
    output:
    '''

    def __init__(self):
        super().__init__()
        self.predictor = Predictor()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(4)

    def forward(self, embedding, x):
        x = x.to(embedding.device)
        parameter = self.predictor(embedding)

        input = x
        x = x.unsqueeze(3)  # [B, # of points, 3, 1]

        # 3 -> 64
        # print(f"x:{x.shape}, w:{parameter['mlp1_w'].shape}, b:{parameter['mlp1_b'].shape}")
        x = parameter['mlp1_w'].matmul(x) + parameter['mlp1_b']
        # batch norm
        x = x.permute([0, 2, 1, 3]).squeeze(3)
        x = self.bn1(x)
        x = x.permute([0, 2, 1]).unsqueeze(3)

        x = x.mul(parameter['mlp1_s']) + parameter['mlp1_b2']
        x = F.relu(x)

        # 64 -> 64
        x = parameter['mlp2_w'].matmul(x) + parameter['mlp2_b']
        x = x.permute([0, 2, 1, 3]).squeeze(3)
        x = self.bn2(x)
        x = x.permute([0, 2, 1]).unsqueeze(3)
        x = x.mul(parameter['mlp2_s']) + parameter['mlp2_b2']
        x = F.relu(x)

        # 64 -> 64
        x = parameter['mlp3_w'].matmul(x) + parameter['mlp3_b']
        x = x.permute([0, 2, 1, 3]).squeeze(3)
        x = self.bn3(x)
        x = x.permute([0, 2, 1]).unsqueeze(3)
        x = x.mul(parameter['mlp3_s']) + parameter['mlp3_b2']
        x = F.relu(x)
        # 64 -> 64
        x = parameter['mlp4_w'].matmul(x) + parameter['mlp4_b']
        x = x.permute([0, 2, 1, 3]).squeeze(3)
        x = self.bn4(x)
        x = x.permute([0, 2, 1]).unsqueeze(3)
        x = x.mul(parameter['mlp4_s']) + parameter['mlp4_b2']

        # 64 -> 64
        x = parameter['mlp5_w'].matmul(x) + parameter['mlp5_b']
        x = x.permute([0, 2, 1, 3]).squeeze(3)
        x = self.bn5(x)
        x = x.permute([0, 2, 1]).unsqueeze(3)
        x = x.mul(parameter['mlp5_s']) + parameter['mlp5_b2']

        # 64 -> 4
        x = parameter['mlp6_w'].matmul(x) + parameter['mlp6_b']
        x = x.mul(parameter['mlp6_s']) + parameter['mlp6_b2']
        # x = F.relu(x)
        x = x.squeeze(3)
        confidence = x[..., [3]]

        return x[..., :3] + input, confidence


class GetTemplate(object):
    def __init__(self, start_from, dataset_train=None, device=None):
        if start_from == "TEMPLATE":
            self.init_template(dataset_train, device)
        elif start_from == "SOUP":
            self.init_soup()
        elif start_from == "TRAININGDATA":
            self.init_trainingdata(dataset_train)
        elif start_from == "SPHERE":
            self.init_sphere(device)
        else:
            print("select valid template type")

    def init_template(self, dataset_train, device):

        self.mesh = trimesh.load(dataset_train, process=False)
        point_set = self.mesh.vertices
        point_set, _, _ = pointcloud_processor.center_bounding_box(point_set)

        self.mesh_HR = trimesh.load(dataset_train, process=False)
        point_set_HR = self.mesh_HR.vertices
        point_set_HR, _, _ = pointcloud_processor.center_bounding_box(point_set_HR)

        vertex = torch.from_numpy(point_set).float()
        self.vertex = vertex.to(device)
        vertex_HR = torch.from_numpy(point_set_HR).float()
        self.vertex_HR = vertex_HR.to(device)

        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.prop = pointcloud_processor.get_vertex_normalised_area(self.mesh)
        assert (np.abs(np.sum(self.prop) - 1) < 0.001), "Propabilities do not sum to 1!)"
        self.prop = torch.from_numpy(self.prop).unsqueeze(0).float()
        self.prop = self.prop.to(device)
        print(f"Using template to initialize template")

    def init_soup(self):
        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh  # Load this anyway to keep access to edge information
        self.vertex = torch.FloatTensor(6890, 3).normal_().cuda()
        self.vertex_HR = self.vertex.clone()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        print(f"Using Random soup to initialize template")

    def init_trainingdata(self, dataset_train=None):
        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh  # Load this anyway to keep access to edge information
        index = np.random.randint(len(dataset_train))
        points = dataset_train.datas[index].squeeze().clone()
        self.vertex = points
        self.vertex_HR = self.vertex.clone()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        print(f"Using training data number {index} to initialize template")

    def init_sphere(self, device, radius=1, num_of_points=6890):

        z = np.random.random(num_of_points)*2*radius - radius
        phi = np.random.random(num_of_points)*2*np.pi

        x = np.sqrt(radius - z**2)*np.cos(phi)
        y = np.sqrt(radius - z**2)*np.sin(phi)
        points = np.stack([x,y,z]).transpose()

        mesh = trimesh.Trimesh(points, faces=np.array([0,1,2]), process=False)
        self.mesh = mesh
        point_set = mesh.vertices
        point_set, _, _ = pointcloud_processor.center_bounding_box(point_set)
        vertex = torch.from_numpy(point_set).float()
        self.vertex = vertex.to(device)
        self.num_vertex = self.vertex.size(0)
        print('using sphere as template')

# class GetTemplate(object):
#     def __init__(self, folder_path=".", dataset_train=None, device=None):
#         assert device is not None
#         self.data_path = folder_path
#         self.device = device
#         self.init_template()
#
#
#     def init_template(self):
#         # if not os.path.exists(os.path.join(self.data_path, "data/template/template.ply")):
#         #     os.system("chmod +x ./data/download_template.sh")
#         #     os.system("./data/download_template.sh")
#         template_folder = os.path.join(self.data_path, 'data/template_set_1')
#         template_files = [os.path.join(template_folder, 'tr_reg_014.ply'),
#                           os.path.join(template_folder, 'tr_reg_028.ply'),
#                           os.path.join(template_folder, 'tr_reg_067.ply'),
#                           os.path.join(template_folder, 'tr_reg_041.ply'),
#                           os.path.join(template_folder, 'tr_reg_000.ply'),
#                           ]
#         template_vertices = []
#         for template in template_files:
#             mesh = trimesh.load(template, process=False)
#             # self.mesh = mesh
#             point_set = mesh.vertices
#             point_set, _, _ = pointcloud_processor.center_bounding_box(point_set)
#             vertices = torch.from_numpy(point_set).float()
#             # vertices = vertices.to(self.device)
#             template_vertices.append(vertices)
#         self.vertex = template_vertices

# print('template file:', template_files)
# print('template number:{}, shape:{}'.format(len(template_vertices), template_vertices[0].shape))


class Meta_Deformation(nn.Module):
    def __init__(self, num_points=6890, bottleneck_size=1024, batch_size=None, dim_template=3,
                 dim_out_patch=3, device=None, template_mode="SPHERE"):
        # this model no point translation, only patch deformation
        super(Meta_Deformation, self).__init__()

        self.patch_deformation = True
        self.point_translation = False

        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        # print(device)
        # self.point_translation = point_translation
        self.dim_template = dim_template
        # self.patch_deformation = patch_deformation
        self.dim_out_patch = dim_out_patch
        self.dim_before_decoder = 3
        self.count = 0
        self.device = device
        # self.start_from = start_from
        # self.dataset_train = dataset_train

        # ------modify------
        self.batch_size = batch_size
        self.encoder = PointNetfeat(num_points, bottleneck_size)
        if template_mode == "TEMPLATE":
            self.template = []
            template_folder = './data/template_set_1'
            template_files = [os.path.join(template_folder, 'tr_reg_014.ply'),
                              os.path.join(template_folder, 'tr_reg_028.ply'),
                              os.path.join(template_folder, 'tr_reg_067.ply'),
                              os.path.join(template_folder, 'tr_reg_041.ply'),
                              os.path.join(template_folder, 'tr_reg_000.ply'),
                              ]
            for filename in template_files:
                temp = GetTemplate(start_from="TEMPLATE", dataset_train=filename, device=device)
                self.template.append(temp)
        elif template_mode == "SPHERE":
            self.template = [GetTemplate(start_from="SPHERE", device=device)]
        self.decoder = get_clones(Meta_Decoder(), N=len(self.template))
        # print(self.decoder)
        # self.template = GetTemplate(device=self.device)
        # for i in range(len(self.template.vertex)):
        #     self.decoder[i] = self.decoder[i].to(device)
        # self.template.vertex[i] = self.template.vertex[i].expand(self.batch_size, -1, -1)
        # self.template.vertex[i] = self.template.vertex[i].permute([0,2,1])

        # self.encoder = PointNetfeat(num_points, bottleneck_size)
        # self.decoder = nn.ModuleList(
        #     [PointGenCon(bottleneck_size=self.dim_before_decoder + self.bottleneck_size)])

    def morph_points(self, embedding, idx=None, temp_num=None):
        template = self.template
        if temp_num is not None:
            num = temp_num
        else:
            num = 0

        if idx is None:  # testing 6890 vertices
            template[num].vertex = template[num].vertex.expand(embedding.shape[0], -1, -1)
        else:  # training 2500 vertex
            # print(template.vertex[i].shape)
            template[num].vertex = template[num].vertex[idx, :]
            template[num].vertex = template[num].vertex.view(embedding.shape[0], -1, 3)
            # template[i].vertex = template[i].vertex.permute([0,2,1])

        morph_points, confidence = self.decoder[num](embedding, template[num].vertex)

        return morph_points

    def decode(self, x, idx=None):
        return self.morph_points(x, idx)

    def forward(self, x, idx=None):

        # assert self.batch_size == x.shape[0]
        embedding = self.encoder(x)
        morph_shapes = []
        morph_shapes_confidence = []
        template = copy.deepcopy(self.template)
        for i in range(len(template)):
            # print(f'{i}, embedding:{embedding.shape}, template:{template[i].vertex.shape}, idx:{idx[0].shape}')
            if idx is None:  # testing 6890 vertices
                template[i].vertex = template[i].vertex.expand(embedding.shape[0], -1, -1)
            else:  # training 2500 vertex
                template[i].vertex = template[i].vertex[idx, :]
                template[i].vertex = template[i].vertex.view(embedding.shape[0], -1, 3)
            morph_points, confidence = self.decoder[i](embedding, template[i].vertex)

            # morph_points.shape = (batch_size, 3, # of points)
            # confidence.shape = (batch_size, 1, # of points)
            # print(confidence.shape)
            morph_shapes.append(morph_points)
            morph_shapes_confidence.append(confidence)
        morph_shapes = torch.stack(morph_shapes, dim=2)
        morph_shapes_confidence = torch.stack(morph_shapes_confidence, dim=2)
        morph_shapes_confidence = F.softmax(morph_shapes_confidence, dim=2)
        morph_shapes = morph_shapes.permute([0, 3, 2, 1])
        morph_shapes_confidence = morph_shapes_confidence.permute([0, 3, 2, 1])
        # morph_shapes.shape = [batch_size, 3, # of templates, # of points]
        # morph_shapes_confidence.shape = [batch_size, 3, # of templates, # of points]
        # print(morph_shapes_confidence.shape)
        # print(morph_shapes.shape)
        # print('h')
        return morph_shapes, morph_shapes_confidence

    def decode_full(self, x):
        outs = []
        div = 20
        batch = int(self.template[0].num_vertex_HR / div)
        for i in range(div - 1):
            rand_grid = self.template[0].template_learned_HR[batch * i:batch * (i + 1)].view(x.size(0), batch,
                                                                                             self.dim_template).transpose(
                1, 2).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.template[0].template_learned_HR[batch * i:].view(x.size(0), -1, self.dim_template).transpose(1,
                                                                                                                      2).contiguous()
        y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def get_points_translation_template(self):
        base_shape = self.template[0].vertex
        if self.patch_deformation:
            base_shape = self.get_patch_deformation_template()[0]
        return [base_shape + self.template[0].vertex_trans]

    def get_patch_deformation_template(self, high_res=False):
        self.eval()
        print("WARNING: the network is now in eval mode!")
        if high_res:
            rand_grid = self.template[0].vertex_HR.transpose(0, 1).contiguous().unsqueeze(0).expand(1,
                                                                                                    self.dim_template,
                                                                                                    -1)
        else:
            rand_grid = self.template[0].vertex.transpose(0, 1).contiguous().unsqueeze(0).expand(1, self.dim_template,
                                                                                                 -1)
        print(rand_grid.shape)
        return [self.decoder[0](rand_grid.permute([0, 2, 1]))]
        # return [self.templateDiscovery[0](rand_grid).squeeze().transpose(1, 0).contiguous()]

    def make_high_res_template_from_low_res(self):
        """
        This function takes a path to the orginal shapenet model and subsample it nicely
        """
        import pymesh
        if not (self.point_translation or self.patch_deformation):
            self.template[0].template_learned_HR = self.template[0].vertex_HR

        if self.patch_deformation:
            templates = self.get_patch_deformation_template(high_res=True)
            self.template[0].template_learned_HR = templates[0]

        if self.point_translation:
            templates = self.get_points_translation_template()

            if self.dim_template == 3:
                template_points = templates[0].cpu().clone().detach().numpy()
                obj1 = pymesh.form_mesh(vertices=template_points, faces=self.template[0].mesh.faces)
                if len(obj1.vertices) < 100000:
                    obj1 = pymesh.split_long_edges(obj1, 0.02)[0]
                    while len(obj1.vertices) < 100000:
                        obj1 = pymesh.subdivide(obj1)
                self.template[0].mesh_HR = obj1
                self.template[0].template_learned_HR = torch.from_numpy(obj1.vertices).cuda().float()
                self.template[0].num_vertex_HR = self.template[0].template_learned_HR.size(0)
                print(f"Make high res template with {self.template[0].num_vertex_HR} points.")
            elif self.dim_template == 2:
                templates = templates[0]
                templates = torch.cat([templates, torch.zeros((templates.size(0), 1)).cuda()], -1)
                template_points = templates.cpu().clone().detach().numpy()
                obj1 = pymesh.form_mesh(vertices=template_points, faces=self.template[0].mesh.faces)
                if len(obj1.vertices) < 100000:
                    obj1 = pymesh.split_long_edges(obj1, 0.02)[0]
                    while len(obj1.vertices) < 100000:
                        obj1 = pymesh.subdivide(obj1)
                self.template[0].mesh_HR = obj1
                self.template[0].template_learned_HR = torch.from_numpy(obj1.vertices).cuda().float()[:,
                                                       :2].contiguous()
                self.template[0].num_vertex_HR = self.template[0].template_learned_HR.size(0)
                print(f"Make high res template with {self.template[0].num_vertex_HR} points.")
            else:
                template_points = templates[0].cpu().clone().detach().numpy()
                self.template[0].mesh_HR = self.template[0].mesh
                self.template[0].template_learned_HR = torch.from_numpy(template_points).cuda().float()
                self.template[0].num_vertex_HR = self.template[0].template_learned_HR.size(0)
                print(f"Can't do high-res because we are in {self.dim_template}D.")

    def save_template_png(self, path):
        print("Saving template...")
        self.eval()
        if self.point_translation:
            templates = self.get_points_translation_template()
            if self.dim_template == 3:
                template_points = templates[0].cpu().clone().detach().numpy()
                mesh_point_translation = trimesh.Trimesh(vertices=template_points,
                                                         faces=self.template[0].mesh.faces, process=False)
                mesh_point_translation.export(os.path.join(path, "mesh_point_translation.ply"))

                p1 = template_points[:, 0]
                p2 = template_points[:, 1]
                p3 = template_points[:, 2]
                fig = plt.figure(figsize=(20, 20), dpi=80)
                fig.set_size_inches(20, 20)
                ax = fig.add_subplot(111, projection='3d', facecolor='white')
                # ax = fig.add_subplot(111, projection='3d',  facecolor='#202124')
                ax.view_init(0, 30)
                ax.set_xlim3d(-0.8, 0.8)
                ax.set_ylim3d(-0.8, 0.8)
                ax.set_zlim3d(-0.8, 0.8)
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')
                ax.scatter(p3, p1, p2, alpha=1, s=10, c='salmon', edgecolor='orangered')
                plt.grid(b=None)
                plt.axis('off')
                fig.savefig(os.path.join(path, "points_" + str(0) + "_" + str(self.count)), bbox_inches='tight',
                            pad_inches=0)
            else:
                print("can't save png if dim template is not 3!")
        if self.patch_deformation:
            templates = self.get_patch_deformation_template()
            if self.dim_template == 3:
                template_points = templates[0].cpu().clone().detach().numpy()
                mesh_patch_deformation = trimesh.Trimesh(vertices=template_points,
                                                         faces=self.template[0].mesh.faces, process=False)
                mesh_patch_deformation.export(os.path.join(path, "mesh_patch_deformation.ply"))
                p1 = template_points[:, 0]
                p2 = template_points[:, 1]
                p3 = template_points[:, 2]
                fig = plt.figure(figsize=(20, 20), dpi=80)
                fig.set_size_inches(20, 20)
                ax = fig.add_subplot(111, projection='3d', facecolor='white')
                # ax = fig.add_subplot(111, projection='3d',  facecolor='#202124')
                ax.view_init(0, 30)
                ax.set_xlim3d(-0.8, 0.8)
                ax.set_ylim3d(-0.8, 0.8)
                ax.set_zlim3d(-0.8, 0.8)
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')
                ax.scatter(p3, p1, p2, alpha=1, s=10, c='salmon', edgecolor='orangered')
                plt.grid(b=None)
                plt.axis('off')
                fig.savefig(os.path.join(path, "deformation_" + str(0) + "_" + str(self.count)),
                            bbox_inches='tight', pad_inches=0)
            else:
                print("can't save png if dim template is not 3!")
        self.count += 1

# if __name__ == '__main__':
#     a = AE_AtlasNet_Humans()
