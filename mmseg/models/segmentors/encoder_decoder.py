import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

# attack
from pdb import set_trace as st
from PIL import Image, ImageEnhance
import numpy as np
# from sklearn.metrics import confusion_matrix  
import kornia

# def compute_iou(y_pred, y_true):
#      # ytrue, ypred is a flatten vector
#      y_pred = y_pred.flatten()
#      y_true = y_true.flatten()
#      current = confusion_matrix(y_true, y_pred, labels=[0, 1])
#      # compute mean iou
#      intersection = np.diag(current)
#      ground_truth_set = current.sum(axis=1)
#      predicted_set = current.sum(axis=0)
#      union = ground_truth_set + predicted_set - intersection
#      IoU = intersection / union.astype(np.float32)
#      return np.mean(IoU)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    
    
    def pgd_t(self, image, label, target_mask, patch_init, patch_orig, img_meta, rescale,
        step_size = 0.1, eps=10/255., iters=10, alpha = 1e-1, beta = 2., restarts=1, 
        target_label=None, rap=False, init_tf_pts=None, patch_mask = None, deeplab=False, log=False):
        
        NORM_MEAN = np.array([0.29010095242892997, 0.32808144844279574, 0.28696394422942517])
        NORM_MEAN = np.array([123.675, 116.28, 103.53])/255.
        NORM_STD = np.array([0.1829540508368939, 0.18656561047509476, 0.18447508988480435])
        NORM_STD = np.array([58.395, 57.12, 57.375])/255.

        images = image
        label = torch.from_numpy(label)
        t_labels = torch.ones_like(label)
        labels = t_labels.cuda()
        patches = patch_init.cuda()

        u_labels = label.cuda()

        # images = torch.autograd.Variable(images)
        # labels = torch.autograd.Variable(labels)
        u_labels = torch.autograd.Variable(u_labels)

        target_mask = torch.from_numpy(target_mask).cuda()

        mean = torch.from_numpy(NORM_MEAN).float().cuda().unsqueeze(0)
        mean = mean[..., None, None]
        std = torch.from_numpy(NORM_STD).float().cuda().unsqueeze(0)
        std = std[..., None, None]

        # loss = nn.CrossEntropyLoss()
        if deeplab:
            loss = nn.CrossEntropyLoss(ignore_index=255)
        else:
            loss = nn.NLLLoss2d(ignore_index=255)

        tv_loss = TVLoss()

        # h_loss = houdini_loss()

        # init transformation matrix
        h, w = images.shape[-2:]  # destination size
        points_src = torch.FloatTensor(init_tf_pts[0]).unsqueeze(0)

        # the destination points are the image vertexes
        points_dst = torch.FloatTensor(init_tf_pts[1]).unsqueeze(0)

        M: torch.tensor = kornia.get_perspective_transform(points_dst, points_src).cuda()

        if patch_mask is None:
            patch_mask_var = torch.ones_like(patches)
        else:
            patch_mask_var = patch_mask
        t_patch_mask_var = kornia.warp_perspective(patch_mask_var.float(), M, dsize=(h, w))

        ori_patches = patch_orig.data

        best_adv_patches = [torch.zeros_like(patches),1e8]


        for j in range(restarts):
            delta = torch.rand_like(patches, requires_grad=True)
            delta = torch.zeros_like(patches, requires_grad=True)
            # delta.data = (delta.data * 2 * eps - eps) * perturb_mask

            # optimizer = torch.optim.Adam([delta], lr=1e-1)

            for i in range(iters) :

                # step_size  = np.max([1e-3, step_size * 0.99])
                step_size  = np.max([1e-3, step_size])
                images.requires_grad = False
                patches.requires_grad = False
                delta.requires_grad = True
                patch_mask_var.requires_grad = False

                # adv_patches = patches + delta
                # eta = torch.clamp(adv_patches - ori_patches, min=-eps, max=eps)
                # adv_patches = torch.clamp(ori_patches + eta, min=0, max=1)

                t_patch: torch.tensor = kornia.warp_perspective((patches+delta).float(), M, dsize=(h, w))
                # t_patch: torch.tensor = kornia.warp_perspective((adv_patches).float(), M, dsize=(h, w))

                adv_images = (torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images*std+mean),min=0, max=1)- mean)/std

                # optimizer.zero_grad()
                self.zero_grad()

                outputs = self.inference(adv_images, img_meta, rescale)

                
            
                # remove attack
                # cost = - loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])

                # rap attack
                if rap:
                    if target_label != None:
                        # target attack
                        obj_loss_value = - loss(outputs*target_mask, labels*target_label*target_mask)
                        tv_loss_value = - tv_loss(ori_patches + delta)
                        cost = - alpha * obj_loss_value - (1-alpha) * tv_loss_value
                    else:
                        # untargeted attack
                        obj_loss_value = loss(outputs*target_mask, u_labels*target_mask)
                        tv_loss_value = - tv_loss(ori_patches + delta)
                        cost = - alpha * obj_loss_value - (1-alpha) * tv_loss_value

                cost.backward()
                # optimizer.step()

                if log:
                    print(i,cost.data, obj_loss_value.data, tv_loss_value.data)

                adv_patches = patches + delta - step_size*eps*delta.grad.sign()
                eta = torch.clamp(adv_patches - ori_patches, min=-eps, max=eps)
                delta = torch.clamp(ori_patches + eta, min=0, max=1).detach_() - patches

                if cost.cpu().data.numpy() < best_adv_patches[1]:
                    best_adv_patches = [delta.data, cost.cpu().data.numpy()]

        t_patch: torch.tensor = kornia.warp_perspective((patches+best_adv_patches[0]).float(), M, dsize=(h, w))

        adv_images = (torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images*std+mean),min=0, max=1)- mean)/std

        return adv_images, best_adv_patches[0]+patches, t_patch_mask_var.cpu().data.numpy()

    def pgd_opt(self, image, label, target_mask, patch_init, patch_orig, img_meta, rescale,
        step_size = 0.1, eps=10/255., iters=10, alpha = 1e-1, beta = 2., restarts=1, 
        target_label=None, rap=False, init_tf_pts=None, patch_mask = None, deeplab=False, log=False):
        
        NORM_MEAN = np.array([0.29010095242892997, 0.32808144844279574, 0.28696394422942517])
        NORM_MEAN = np.array([123.675, 116.28, 103.53])/255.
        NORM_STD = np.array([0.1829540508368939, 0.18656561047509476, 0.18447508988480435])
        NORM_STD = np.array([58.395, 57.12, 57.375])/255.


        images = image
        label = torch.from_numpy(label)
        t_labels = torch.ones_like(label)
        labels = t_labels.cuda()
        patches = patch_init.cuda()

        u_labels = label.cuda()

        # images = torch.autograd.Variable(images)
        # labels = torch.autograd.Variable(labels)
        u_labels = torch.autograd.Variable(u_labels)

        target_mask = torch.from_numpy(target_mask).cuda()

        mean = torch.from_numpy(NORM_MEAN).float().cuda().unsqueeze(0)
        mean = mean[..., None, None]
        std = torch.from_numpy(NORM_STD).float().cuda().unsqueeze(0)
        std = std[..., None, None]

        # loss = nn.CrossEntropyLoss()
        if deeplab:
            loss = nn.CrossEntropyLoss(ignore_index=255)
        else:
            loss = nn.NLLLoss2d(ignore_index=255)

        tv_loss = TVLoss()

        # h_loss = houdini_loss()

        best_adv_img = [torch.zeros_like(images.data), -1e8]

        # init transformation matrix
        h, w = images.shape[-2:]  # destination size
        points_src = torch.FloatTensor(init_tf_pts[0]).unsqueeze(0)

        # the destination points are the image vertexes
        points_dst = torch.FloatTensor(init_tf_pts[1]).unsqueeze(0)

        M: torch.tensor = kornia.get_perspective_transform(points_dst, points_src).cuda()

        if patch_mask is None:
            patch_mask_var = torch.ones_like(patches)
        else:
            patch_mask_var = patch_mask
        t_patch_mask_var = kornia.warp_perspective(patch_mask_var.float(), M, dsize=(h, w))

        ori_patches = patch_orig.data

        best_adv_patches = [torch.zeros_like(patches),1e8]


        for j in range(restarts):
            delta = torch.rand_like(patches, requires_grad=True)
            delta = torch.zeros_like(patches, requires_grad=True)
            # delta.data = (delta.data * 2 * eps - eps) * perturb_mask

            # optimizer = torch.optim.Adam([delta], lr=step_size)
            optimizer = torch.optim.AdamW([delta], lr=step_size, weight_decay=0.01)

            for i in range(iters) :

                # step_size  = np.max([1e-3, step_size * 0.99])
                step_size  = np.max([1e-3, step_size])
                images.requires_grad = False
                patches.requires_grad = False
                delta.requires_grad = True
                patch_mask_var.requires_grad = False

                adv_patches = patches + delta
                eta = torch.clamp(adv_patches - ori_patches, min=-eps, max=eps)
                adv_patches = torch.clamp(ori_patches + eta, min=0, max=1)

                t_patch: torch.tensor = kornia.warp_perspective((adv_patches).float(), M, dsize=(h, w))

                adv_images = (torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images*std+mean),min=0, max=1)- mean)/std

                optimizer.zero_grad()
                self.zero_grad()

                outputs = self.inference(adv_images, img_meta, rescale)

                # rap attack
                if rap:
                    if target_label != None:
                        # target attack
                        obj_loss_value = - loss(outputs*target_mask, labels*target_label*target_mask)
                        tv_loss_value = - tv_loss(ori_patches + delta)
                        cost = - alpha * obj_loss_value - (1-alpha) * tv_loss_value
                    else:
                        # untargeted attack
                        obj_loss_value = loss(outputs*target_mask, u_labels*target_mask)
                        tv_loss_value = - tv_loss(ori_patches + delta)
                        cost = - alpha * obj_loss_value - (1-alpha) * tv_loss_value

                cost.backward()
                optimizer.step()

                if log:
                    print(i,cost.data, obj_loss_value.data, tv_loss_value.data)

                # adv_patches = patches + delta - step_size*eps*delta.grad.sign()
                # eta = torch.clamp(adv_patches - ori_patches, min=-eps, max=eps)
                # delta = torch.clamp(ori_patches + eta, min=0, max=1).detach_() - patches

                if cost.cpu().data.numpy() < best_adv_patches[1]:
                    best_adv_patches = [delta.data, cost.cpu().data.numpy()]

        t_patch: torch.tensor = kornia.warp_perspective((patches+best_adv_patches[0]).float(), M, dsize=(h, w))

        adv_images = (torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images*std+mean),min=0, max=1)- mean)/std

        return adv_images, best_adv_patches[0]+patches, t_patch_mask_var.cpu().data.numpy()
    
    def simple_attack(self, img, img_meta,gt_semantic_seg, rescale=True):
        """Simple test with single image."""

        with torch.no_grad():
            seg_logit = self.inference(img, img_meta, rescale)
            seg_pred = seg_logit.argmax(dim=1)
            label = seg_pred.cpu().numpy()
        
        return list(label), img


        # calculate ERF
        # img.requires_grad = True
        # seg_logit = self.inference(img, img_meta, rescale)
        # cost = seg_logit[0,0,512,1024]
        # cost.backward()

        # grad_img = np.moveaxis(img.grad.cpu().numpy()[0],0,-1)


        label = gt_semantic_seg[0].cpu().numpy().astype(np.long)

        # return list(label), img, grad_img


        # st()
        # if torch.onnx.is_in_onnx_export():
        #     # our inference backend only support 4D output
        #     seg_pred = seg_pred.unsqueeze(0)
        #     return seg_pred
        # label = np.ones([1,1024,2048],dtype=np.long)
        
        # torch.cuda.empty_cache()
        # rap attack
        
        target_labels = [
            5,6,7, # object: pole, traffic light, traffic sign
            11,12, # human: person, rider
            13,14,15,16,17,18 # vehicle: car, truck, bus, train, motorcycle, bicycle
        ]
        
        h, w = img.size()[2:4]

        init_tf_pts = np.array([
                # [[0, h-300], [300 - 1, h-300], [300 - 1, h - 1], [0, h - 1]],
                [[928, 574],[1205, 574],[1262, 663],[851, 664]], # small
                # [[970, 507],[1161, 507],[1287, 664],[851, 664]], # large
                [[0, 0], [300 - 1, 0], [300 - 1, 300 - 1], [0, 300 - 1]],
                # [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
            ]).astype(np.int)

        shift = 0
        init_tf_pts[0][:,0]+=shift

        patch_im = Image.open('phy_exp/cropped_patch.jpg')
        patch_img = np.zeros_like(img.cpu())
        p_img = np.array(patch_im.resize((300,300)))/255.
        p_img = np.moveaxis(p_img,-1,0)
        # p_img = np.ones_like(p_img) * 0.5
        patch_img[0,:,:300,:300] = p_img

        patch_orig = torch.from_numpy(patch_img).cuda()
        adv_patch = torch.from_numpy(patch_img).cuda()

        patch_mask = np.zeros_like(patch_img)
        patch_mask[0,:,:300,:300] = 1
        patch_mask = torch.from_numpy(patch_mask).cuda()
        
        target_mask = np.zeros_like(label)
        # target_mask[:,300:800,800:1300] = 1
        target_mask[:,int(h/2-200):int(h/2+200),int(w/2-200):int(w/2+200)] = 1
        # target_mask = np.ones_like(label)
        eval_target_mask = target_mask.copy()
        target_mask = (np.any([label == id for id in target_labels],axis = 0) & (target_mask == 1)).astype(np.long) 
        target_mask = target_mask.astype(np.int8) 
        loss_mask = target_mask.copy()

        if target_mask.sum() < 1e-8:
            adv_image = img
        else:
            adv_image, adv_patch = self.pgd_opt(img,label,loss_mask,adv_patch,patch_orig, img_meta, rescale,
                        init_tf_pts=init_tf_pts, 
                        step_size = 1e-2, eps=0./255, iters=1, 
                        target_label = 2,
                        deeplab=True,
                        alpha=1, beta=1, restarts=1, rap=True,  patch_mask=patch_mask, log=True)[:2]
        
        
        seg_logit = self.inference(adv_image, img_meta, rescale)
        adv_label = seg_logit.argmax(dim=1)
        adv_label = adv_label.cpu().numpy()
        
        # unravel batch dim
        adv_label = list(adv_label)
        return adv_label, adv_image

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
