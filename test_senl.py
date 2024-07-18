from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import RGBN300Data,TestData
from data_manager import *
from eval_metrics import reideval,evaluate_rank
from utils import *
from tensorboardX import SummaryWriter
from confignl import argument_parser
from loss import pdist_np
# testing

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    if args.dataset == 'rgbn300':
        dataset = 'rgbn300'
        data_path = '/media/jqzhu/941A7DD31A7DB33A/ZQQ/datasets/RGBN300_autolabel'
    else:
        print('no dataset')

    log_path = args.log_path + '/' + dataset

    attnum=0
    for i in range(len(args.attloc)):
        attnum = attnum +  args.attloc[i]

    if attnum==0:
        print('none attenion msew=0')
        args.msew=0

    suffix = '{}_p{}_n{}_msew{}_w{}_h{}_epoch{}'.format(args.attloc,args.num_pos, args.batch_size,args.msew,args.imgw,args.imgh,args.epochcount)
    log_path = log_path + '/' + suffix + '/'
    print('log_path')
    print(log_path)

    checkpoint_path = log_path
    if not os.path.isdir(log_path):
        os.makedirs(log_path)


    from model_senl import single50_senl
    net = single50_senl(class_num=150,attloc=args.attloc)
    if len(args.gpu)>1:
        net=torch.nn.DataParallel(net).cuda()#.to(device)
    else:
        net=net.cuda()

    net.eval()
    model_path = checkpoint_path + args.modelid
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    print('==> loaded model: {})'.format(model_path))


    out_path = log_path  + args.mode
    print(out_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    sys.stdout = Logger(out_path + '/' + 'log.txt')
    writer = SummaryWriter(out_path)

    trial_num = 1
    for i in range(1, trial_num + 1):
        args.trial = i

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.imgh, args.imgw)),
            transforms.ToTensor(),
            normalize,
        ])

        # testing set
        query_img, query_label, query_cam, gallery_img, gallery_label, gallery_cam = \
            process_eval_rgbnt(data_path, eval_mode=args.mode, imgh=args.imgh, imgw=args.imgw)

        galleryset = TestData(gallery_img, gallery_label, transform=transform_test, img_size=(args.imgw, args.imgh))
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.imgw, args.imgh))

        # testing data loader
        gallery_loader = data.DataLoader(galleryset, batch_size=args.test_batch, shuffle=False,
                                         num_workers=args.workers)
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gallery_label)

        print('Dataset {} statistics:'.format(dataset))
        print('  ------------------------------')
        print('  subset   | # ids | # images')
        print('  ------------------------------')
        print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
        print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gallery_label)), ngall))
        print('  ------------------------------')
        print('==> Building model..')

        cmc_att, mAP_att, mINP_att = test(net, writer, dataset,
                                          ngall, gallery_loader, gallery_label, gallery_cam,
                                          nquery, query_loader, query_label, query_cam
                                          )
        # print(args.mode, args.trial, cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att)

        # print(args.mode, args.trial, cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att)

        if i == 1:
            cmc_att_all = cmc_att
            mAP_att_all = mAP_att
            mINP_att_all = mINP_att
        else:
            cmc_att_all = cmc_att_all + cmc_att
            mAP_att_all = mAP_att_all + mAP_att
            mINP_att_all = mINP_att_all + mINP_att

    cmc_att_all = cmc_att_all / trial_num
    mAP_att_all = mAP_att_all / trial_num
    mINP_att_all = mINP_att_all / trial_num

    print('average:', cmc_att_all[0], cmc_att_all[4], cmc_att_all[9], cmc_att_all[19], mAP_att_all, mINP_att_all)

    print('Pool + FC:  for CMC curvers : ')
    for i in range(20):
        print(cmc_att_all[i])




def test(net,writer, dataset,
          ngall, gall_loader, gall_label, gall_cam, nquery, query_loader, query_label, query_cam):

    cmc_att, mAP_att, mINP_att = eval(
        0,
        writer,
        dataset,
        net,
        ngall, gall_loader, gall_label, gall_cam,
        nquery, query_loader, query_label, query_cam
    )
    return cmc_att, mAP_att, mINP_att

def eval(epoch,writer ,dataset,net,ngall,gall_loader,
         gall_label,gall_cam,nquery,query_loader,
         query_label,query_cam):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')

    ptr = 0
    # dim=2048
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att = net(input, input, -0.5)
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num

    net.eval()
    ptr = 0
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att = net(input, input, 0.5)
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num

    if args.mode=='v2t':
        print('...................... v2t eval..................')
    elif args.mode=='t2v':
        print('...................... t2v eval..................')
    elif args.mode=='mix':
        print('...................... mix eval..................')
    else:
        print('.....................wrong mode..................')
        assert 1==0

    distmat_att = pdist_np(query_feat_att, gall_feat_att)
    cmc_att, mAP_att, mINP_att = evaluate_rank(distmat_att, query_label, gall_label, query_cam, gall_cam)

    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)

    return cmc_att, mAP_att, mINP_att


if __name__ == '__main__':
    cudnn.benchmark = True
    # test = torch.Tensor([[1,2,3],[4,5,6]])
    # print(test.view(-1))
    parser = argument_parser()
    args = parser.parse_args()
    main(args)