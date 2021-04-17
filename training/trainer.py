import torch
import torch.optim as optim
import time
import my_utils
import model
import extension.get_chamfer as get_chamfer
import dataset
from termcolor import colored
from abstract_trainer import AbstractTrainer
import os

import pdb



class Trainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.git_repo_path = ""
        self.init_save_dict(opt)
        self.dataset_train = None

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        # model=nn.DataParallel(model,device_ids=[0,1,2]) # multi-GPU
        network = model.Meta_Deformation(dim_template=self.opt.dim_template,
                                        dim_out_patch=self.opt.dim_out_patch,
                                        batch_size=self.opt.batch_size,
                                        device=self.device,
                                        template_mode = self.opt.template_mode,
                                        )
        network=network.to(self.device)  # put network on GPU
        network = torch.nn.DataParallel(network, device_ids=[0])

        network.apply(my_utils.weights_init)  # initialization of the weight
        if self.opt.model != "":
            try:
                network.load_state_dict(torch.load(self.opt.model))
                print(" Previous network weights loaded! From ", self.opt.model)
            except:
                print("Failed to reload ", self.opt.model)
        if self.opt.reload:
            print(f"reload model frow :  {self.opt.dir_name}/network.pth")
            self.opt.model = os.path.join(self.opt.dir_name, "network.pth")
            network.load_state_dict(torch.load(self.opt.model))

        # network = torch.nn.DataParallel(network.cuda(), device_ids=[0,1,2,3])

        # for single gpu
        network = network.to(self.device)

        self.network = network
        self.network.eval()
        # self.network.save_template_png(self.opt.dir_name)
        # self.network.train()

    def _loss_(self, morph_shapes, morph_shapes_confidence, target, mseloss_factor=0.5, Train=False):
        # mean square loss
        mseloss = morph_shapes - target.unsqueeze(2)
        if Train:
            mseloss = mseloss ** 2
            mseloss = torch.mean(torch.mean(mseloss, dim=2))
        else:
            mseloss = morph_shapes_confidence * (mseloss ** 2)
            mseloss = torch.mean(torch.sum(mseloss, dim=2))
        aggr_shape = torch.sum(torch.mul(morph_shapes, morph_shapes_confidence), dim=2)
        aggr_loss = (aggr_shape - target) ** 2
        aggr_loss = torch.mean(aggr_loss)
        loss = mseloss * mseloss_factor + aggr_loss * (1 - mseloss_factor)
        return mseloss, aggr_loss, loss

    def build_optimizer(self):
        """
        Create optimizer
        """
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.opt.reload:
            self.optimizer.load_state_dict(torch.load(f'{self.opt.checkpointname}'))
            my_utils.yellow_print("Reloaded optimizer")

    def build_dataset_train(self):
        """
        Create training dataset
        """
        self.dataset_train = dataset.SURREAL(train=True, regular_sampling=True)
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.opt.batch_size,
                                                            shuffle=True, num_workers=int(self.opt.workers),
                                                            drop_last=True)
        self.len_dataset = len(self.dataset_train)

    def build_dataset_test(self):
        """
        Create testing dataset
        """
        self.dataset_test = dataset.SURREAL(train=False)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=5,
                                                           shuffle=False, num_workers=int(self.opt.workers),
                                                           drop_last=True)
        self.len_dataset_test = len(self.dataset_test)

    def build_losses(self):
        """
        Create losses
        """
        self.distChamfer = get_chamfer.get(self.opt)

    def train_iteration(self):
        self.optimizer.zero_grad()

        pointsReconstructed, morph_confidence = self.network(self.points, self.idx)  # forward pass # [batch, num_point, 3]
        # loss_train_total = torch.mean(
        #         (pointsReconstructed.view(self.points.size(0), -1, 3) - self.points.transpose(2, 1).contiguous()) ** 2)
        # loss_train_total.backward()
        # print("output", pointsReconstructed.shape)
        # print("points", self.points.shape)
        mseloss, aggr_loss, loss = self._loss_(pointsReconstructed, morph_confidence, self.points, Train=True)
        loss.backward()
        self.log.update("loss_train_total", loss.item())
        self.optimizer.step()  # gradient update

        # VIZUALIZE
        # if self.iteration % 100 == 1 and self.opt.display:
        #     pointsReconstructed_mean = torch.mean(pointsReconstructed, dim=2)
        #     pointsReconstructed_mean = pointsReconstructed_mean.permute([0,2,1])
        #     self.visualizer.show_pointclouds(points=self.points[0], title="train_input")
        #     self.visualizer.show_pointclouds(points=pointsReconstructed_mean[0], title="train_input_reconstructed")
        #     if self.opt.dim_template == 3:
        #         self.visualizer.show_pointclouds(points=self.network.template[0].vertex, title=f"template0")
        #     if self.opt.patch_deformation and self.opt.dim_out_patch == 3:
        #         template = self.network.get_patch_deformation_template()
        #         self.network.train() #Add this or the training keeps going in eval mode!
        #         print("Network in TRAIN mode!")
        #         self.visualizer.show_pointclouds(points=template[0], title=f"template_deformed0")
        self.print_iteration_stats(loss)

    def optim_reset(self, flag):
        if flag:
            # Currently I choose to reset the optimiser because it's to complicated to copy the branch optims
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.lrate) # get new optimiser
            # optimizer.load_state_dict(self.optimizer.state_dict()) # copy state
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.opt.lrate)

    def train_epoch(self):
        self.log.reset()
        self.network.train()
        self.learning_rate_scheduler()
        start = time.time()
        iterator = self.dataloader_train.__iter__()
        self.reset_iteration()
        while True:
            try:
                # if self.iteration > 10:
                #     break
                points, idx, _, _ = iterator.next()
                points = points.transpose(2, 1).contiguous()
                points = points.to(self.device)
                self.points = points
                self.idx = idx
                # print(self.points.shape)
                self.increment_iteration()
            except:
                print(colored("end of train dataset", 'red'))
                break
            self.train_iteration()
        print("Ellapsed time : ", time.time() - start)

    def test_iteration(self):
        pointsReconstructed, morph_confidence = self.network(self.points)  # forward pass # [batch, num_point, 3]

        l2_loss, _, loss = self._loss_(pointsReconstructed, morph_confidence, self.points)


        self.log.update("loss_val_Deformation_ChamferL2", l2_loss)
        print(
            '\r' + colored('[%d: %d/%d]' % (self.epoch, self.iteration, self.len_dataset_test / (self.opt.batch_size)),
                           'red') +
            colored('loss_val_Deformation_ChamferL2:  %f' % loss.item(), 'yellow'),
            end='')

        # if self.iteration % 60 == 1 and self.opt.display:
        #     self.visualizer.show_pointclouds(points=self.points[0], title="test_input")
        #     self.visualizer.show_pointclouds(points=pointsReconstructed[0], title="test_input_reconstructed")

    def test_epoch(self):
        self.network.eval()
        iterator = self.dataloader_test.__iter__()
        self.reset_iteration()
        while True:
            self.increment_iteration()
            try:
                # if self.iteration > 10:
                #     break
                points, _, _, _ = iterator.next()
                points = points.transpose(2, 1).contiguous()
                points = points.to(self.device)
                self.points = points
            except:
                print(colored("end of val dataset", 'red'))
                break
            self.test_iteration()

        self.log.end_epoch()
        if self.opt.display:
            self.log.update_curves(self.visualizer.vis, self.opt.dir_name)
            # self.network.save_template_png(self.opt.dir_name)
