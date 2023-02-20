import torch, numpy as np, os, pandas as pd, scipy.io as scio, torchvision.transforms as transforms, torchvision
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from PIL import Image


class myDataLoader(object):
    def __init__(self, opts) -> None:
        self.opts = opts
        self.device = opts.device
        # load dataset
        if opts.task == 'sketching':
            self.load_dataset_sketching(opts.dataset)
        elif opts.task == 'morphing':
            self.load_dataset_morphing(opts.dataset)
        elif opts.task == 'color_transfer':
            self.load_dataset_ct(opts.dataset)
        elif opts.task == 'flow':
            self.load_dataset_flow(opts.dataset)
        else:
            raise NotImplementedError

    def load_dataset_sketching(self, dataset_name):
        main_dir = os.path.join('./datasets', 'sketching')
        if dataset_name in ['cheetah']:
            img_path = os.path.join(main_dir, dataset_name, 'target.jpg')
            image = Image.open(img_path)
            self.aspect_hw = 1.0 * image.height / image.width
            pix = np.array(image)
            self.original = pix
            if dataset_name in ['cheetah']:
                pix = pix[:, :, 0]
                pix = 255 - pix
            # create a meshgrid and interpret the image as a probability distribution on it
            x_grid = torch.linspace(0, 1, steps = pix.shape[0]) # x is the height and y is the width, we will convert later
            y_grid = torch.linspace(0, pix.shape[1] / pix.shape[0], steps = pix.shape[1])
            x_mesh, y_mesh = torch.meshgrid(x_grid, y_grid, indexing = 'ij')
            x_mesh = x_mesh.reshape(-1)
            y_mesh = y_mesh.reshape(-1)
            pix_arr = pix.reshape(-1)
            tgt_support = []
            tgt_mass = []
            if dataset_name in ['cheetah']:
                value_thr = 50
            else:
                value_thr = 0
            for value, x, y in zip(pix_arr, x_mesh, y_mesh):
                if value > value_thr:
                    tgt_support.append(torch.tensor([y, 1 - x]))
                    tgt_mass.append(torch.tensor(value, dtype = torch.float32))
            
            self.tgt_support = torch.stack(tgt_support, dim = 0).to(self.device)
            tgt_mass = torch.stack(tgt_mass, dim = 0).to(self.device)
            self.tgt_mass = tgt_mass / tgt_mass.sum()
        else:
            raise NotImplementedError

    def load_dataset_morphing(self, dataset_name):
        main_dir = './datasets/morphing'
        if dataset_name == 'gaussian2circle':
            raise NotImplementedError
            self.tgt_support = 0.9 * generate_circles_data(self.opts.max_tgt_num, noise = 0.05, factor = 0.5)[0]
            self.tgt_support = torch.from_numpy(self.tgt_support).to(self.device).contiguous()
            self.tgt_mass = torch.ones(self.tgt_support.shape[0], device = self.device, dtype = self.tgt_support.dtype)
            self.tgt_mass = self.tgt_mass / self.tgt_mass.sum()
            self.src_support = 0.1 * torch.randn(size = (self.opts.particle_num, self.tgt_support.shape[1]), device = self.device, dtype = self.tgt_support.dtype)
            self.src_mass = torch.ones(self.opts.particle_num, device = self.device, dtype = self.src_support.dtype)
            self.src_mass = self.src_mass / self.src_mass.sum()
        elif '[]' in dataset_name:
            source_name = dataset_name[: dataset_name.find('[]')]
            target_name = dataset_name[dataset_name.find('[]') + 2:]
            image_names = os.listdir(main_dir)
            if (source_name not in image_names) or (target_name not in image_names):
                raise RuntimeError('image {} can not be found in datasets/morphing'.format(image_names))
            self.src_support = torch.from_numpy(discretize_image(os.path.join(main_dir, source_name), resize = self.opts.resize, max_num = self.opts.max_src_num)).to(self.device).contiguous()
            self.tgt_support = torch.from_numpy(discretize_image(os.path.join(main_dir, target_name), resize = self.opts.resize, max_num = self.opts.max_tgt_num)).to(self.device).contiguous()
            self.src_mass = torch.ones(self.src_support.shape[0], device = self.device, dtype = self.src_support.dtype) / self.src_support.shape[0]
            if not self.opts.hgradient:
                self.tgt_mass = torch.ones(self.tgt_support.shape[0], device = self.device, dtype = self.tgt_support.dtype) / self.tgt_support.shape[0]
            else:
                support_int = (self.tgt_support * self.opts.resize).detach().cpu().numpy().astype(np.int)
                # reverse the y axis
                support_int[:, 1] = self.opts.resize - support_int[:, 1] - 1
                weight_image = np.array(Image.open(os.path.join(main_dir, 'hgradient.png')).resize((self.opts.resize, self.opts.resize)).convert('L'))
                mass = weight_image[support_int[:, 1], support_int[:, 0]]
                mass = (mass - min(mass)) / (max(mass) - min(mass)) + 0.1
                self.tgt_mass = torch.from_numpy(mass).to(self.device).contiguous()
                self.tgt_mass = self.tgt_mass / self.tgt_mass.sum()
        else:
            raise NotImplementedError

    def load_dataset_ct(self, dataset_name):
        main_dir = os.path.join('./datasets', 'color_transfer', dataset_name)
        src_image = Image.open(os.path.join(main_dir, 'source.jpg')).convert('RGB')
        tgt_image = Image.open(os.path.join(main_dir, 'target.jpg')).convert('RGB')
        if  isinstance(self.opts.square_size, int) and self.opts.square_size > 0:
            src_image = src_image.resize((self.opts.square_size, self.opts.square_size), Image.Resampling.BICUBIC)
            tgt_image = tgt_image.resize((self.opts.square_size, self.opts.square_size), Image.Resampling.BICUBIC)
        self.image_size = src_image.height
        src_transforms = transforms.Compose([
            transforms.Resize((self.image_size // self.opts.src_downscale, self.image_size // self.opts.src_downscale)),
            transforms.ToTensor()
        ])
        tgt_transforms = transforms.Compose([
            transforms.Resize((self.image_size // self.opts.tgt_downscale, self.image_size // self.opts.tgt_downscale)),
            transforms.ToTensor()
        ])
        self.src_image, self.tgt_image = src_transforms(src_image), tgt_transforms(tgt_image)
        self.src_support = self.src_image.view(3, -1).transpose(0, 1).to(self.device).contiguous()
        self.src_mass = torch.ones(self.src_support.shape[0], device = self.device, dtype = self.src_support.dtype).contiguous()
        self.src_mass = self.src_mass / self.src_mass.sum()
        self.tgt_support = self.tgt_image.view(3, -1).transpose(0, 1).to(self.device).contiguous()
        self.tgt_mass = torch.ones(self.tgt_support.shape[0], device = self.device, dtype = self.tgt_support.dtype).contiguous()
        self.tgt_mass = self.tgt_mass / self.tgt_mass.sum()

    def load_dataset_flow(self, dataset_name):
        main_dir = os.path.join('./datasets', 'flow',dataset_name)
        if dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(
                root = main_dir, train = True, transform = None, download = True
            ).data.transpose((0, 3, 1, 2))[:self.opts.tgt_num]
            self.image_size = (32, 32)
            self.tgt_support = torch.from_numpy(dataset).to(self.device).contiguous().view(len(dataset), -1) / 255
            self.tgt_mass = torch.ones(self.tgt_support.shape[0], device = self.device, dtype = self.tgt_support.dtype)
            self.tgt_mass = self.tgt_mass / self.tgt_mass.sum()
        else:
            raise RuntimeError('No dataset {}'.format(dataset_name))



def discretize_image(image_path, resize: int, max_num: int) -> np.ndarray:
    image = np.array(Image.open(image_path).resize((resize, resize)).convert('L'))
    y_inv, x = np.nonzero(image <= 128)
    y = resize - y_inv - 1
    if max_num > 0 and x.size > max_num:
        index = np.random.choice(x.size, max_num, replace = False)
        x, y = x[index], y[index]
    return np.stack((x, y), axis = 1) / resize

def generate_circles_data(n_samples = 1000, noise = None, factor=.8):
    """Make a large circle containing a smaller circle in 2d.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Total number of points for both circles
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    factor : 0 < double < 1 (default=.8)
        Scale factor between inner and outer circle.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    if factor >= 1 or factor < 0:
        raise ValueError("'factor' has to be between 0 and 1.")

    n_samples_outer = (n_samples + 1) // 2
    n_samples_inner = n_samples // 2

    linspace_outer = np.linspace(0, 2 * np.pi, n_samples_outer, endpoint = False)
    linspace_inner = np.linspace(0, 2 * np.pi, n_samples_inner, endpoint = False)
    outer_circ_x = np.cos(linspace_outer)
    outer_circ_y = np.sin(linspace_outer)
    inner_circ_x = np.cos(linspace_inner) * factor
    inner_circ_y = np.sin(linspace_inner) * factor

    X = np.vstack([np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_outer, dtype = np.intp), np.ones(n_samples_inner, dtype=np.intp)])

    if noise is not None:
        X += np.random.normal(0.0, noise, size = X.shape)

    return X, y
    
'''
    def get_train_loader(self):
        # set dataloader, 
        if self.opts.task == 'logistic_regression': # fake dataloader, keep code consistent
            self.train_loader = [[0, 0]]
        elif self.opts.task in ['ica', 'ica_meg']:
            self.train_loader = [[0, 0]]
        elif self.opts.task == 'gaussian_process':
            self.train_loader = [[0, 0]]
        elif self.opts.task in ['bnn_regression', 'bnn_classification']:
            train_set = TensorDataset(self.train_features, self.train_labels)
            self.train_loader = DataLoader(
                dataset = train_set,
                batch_size = self.opts.batch_size,
                shuffle = True,
                drop_last = True
            )
        elif self.opts.task in ['funnel', 'single_gaussian', 'multi_gaussian', 'demo']:
            self.train_loader = [[0, 0]]
        else:
            raise NotImplementedError
        return self.train_loader

    def load_dataset_gp(self, dataset_name):
        data_file = os.path.join('datasets', dataset_name, dataset_name + '.mat')
        raw_data = scio.loadmat(data_file)
        self.train_features = torch.from_numpy(raw_data['range'].astype(np.float32)).to(self.device) # N * 1 array
        self.train_labels = torch.from_numpy(raw_data['logratio'].astype(np.float32)).to(self.device) # N * 1 array
        self.test_features = None
        self.test_labels = None
        self.model_dim = 2
        self.train_num = self.train_features.shape[0]

    def load_dataset_meg(self, dataset_name):
        lines = []
        with open(os.path.join('datasets', dataset_name, dataset_name + '.txt')) as f:
            for i in range(self.opts.dim):
                lines.append(list(map(float,f.readline().strip().split('  '))))
        data_raw = np.array(lines)
        train_features, test_features = train_test_split(data_raw.transpose(), train_size = self.opts.train_num, random_state = self.opts.split_seed)
        self.train_features = torch.from_numpy(train_features).float().to(self.device)
        self.test_features = torch.from_numpy(test_features).float().to(self.device)
        self.model_dim = self.opts.dim ** 2
        self.train_num = self.opts.train_num
        self.test_num = self.test_features.shape[0]
        self.train_labels = torch.zeros(self.train_num, device = self.device) # fake labels
        #self.test_features /= 1000
        #self.train_features /= 1000
        std_features = torch.std(self.train_features, dim = 0, unbiased = True)
        std_features[std_features == 0] = 1
        mean_features = torch.mean(self.train_features, dim = 0)
        self.train_features = (self.train_features - mean_features) / std_features
        self.test_features = (self.test_features - mean_features) / std_features

    def load_dataset_cl(self, dataset_name):
        main_folder = os.path.join('datasets')
        if dataset_name in 'usps': 
            train_path = os.path.join(main_folder, dataset_name, dataset_name + '-train.txt')
            test_path = os.path.join(main_folder, dataset_name, dataset_name + '-test.txt')
            train_labels_raw, train_features_raw = svm_read_problem(train_path, return_scipy = True)
            test_labels_raw, test_features_raw = svm_read_problem(test_path, return_scipy = True)
            if dataset_name in 'usps': 
                train_labels_raw = train_labels_raw - 1
                test_labels_raw = test_labels_raw - 1
            self.num_classes = int(train_labels_raw.max() + 1)
        else: 
            raise ValueError('dataset {} not found'.format(dataset_name))
        train_features_raw = train_features_raw.toarray()
        test_features_raw = test_features_raw.toarray()
        # to Tensor
        self.train_features = torch.from_numpy(train_features_raw).float().to(self.device)
        self.train_labels =  torch.nn.functional.one_hot(torch.from_numpy(train_labels_raw).long(), self.num_classes).to(self.device)
        self.test_features = torch.from_numpy(test_features_raw).float().to(self.device)
        self.test_labels =  torch.nn.functional.one_hot(torch.from_numpy(test_labels_raw).long(), self.num_classes).to(self.device)
        # record information
        self.data_dim = len(self.train_features[0])
        self.train_num = len(self.train_features)
        self.test_num = len(self.test_features)
        self.model_dim = self.data_dim * self.n_hidden + self.n_hidden * self.num_classes +\
            self.n_hidden + self.num_classes
        

    def load_dataset_lr(self, dataset_name):
        main_folder = os.path.join('datasets')
        # load and split dataset
        if dataset_name in 'a3a, a9a, ijcnn, gisette, w8a, a8a, codrna, madelon':  # no need to split dataset
            train_path = os.path.join(main_folder, dataset_name, dataset_name + '-train.txt')
            test_path = os.path.join(main_folder, dataset_name, dataset_name + '-test.txt')
            train_labels_raw, train_features_raw = svm_read_problem(train_path, return_scipy = True)
            test_labels_raw, test_features_raw = svm_read_problem(test_path, return_scipy = True)
        elif dataset_name in 'mushrooms, pima, covtype, phishing, susy, fourclass, heart':    # split dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.txt')
            labels_raw, features_raw = svm_read_problem(data_path, return_scipy = True)
            if dataset_name in 'mushrooms, covtype': labels_raw = (labels_raw - 1.5) * 2
            if dataset_name in 'phishing, susy': labels_raw = (labels_raw - 0.5) * 2
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)
        else:
            raise ValueError('dataset {} not found'.format(dataset_name))
        # some extra process for certain dataset
        train_features_raw = train_features_raw.toarray()
        test_features_raw = test_features_raw.toarray()
        self.train_num = len(train_labels_raw)
        self.test_num = len(test_labels_raw)
        if dataset_name == 'a3a':
            train_features_raw = np.concatenate((train_features_raw, np.zeros([self.train_num,1])), axis=1)
        if dataset_name in 'a9a, a8a':
            test_features_raw = np.concatenate((test_features_raw, np.zeros([self.test_num,1])), axis=1)
        # to Tensor
        self.train_features = torch.from_numpy(train_features_raw).float().to(self.device)
        self.train_labels = torch.from_numpy(train_labels_raw).float().to(self.device)
        self.test_features = torch.from_numpy(test_features_raw).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels_raw).float().to(self.device)
        self.data_dim = len(self.train_features[0])
        self.model_dim = self.data_dim + 1 # + 1 for bias
        # scale
        std_features = torch.std(self.train_features, dim = 0, unbiased = True)
        std_features[std_features == 0] = 1
        mean_features = torch.mean(self.train_features, dim = 0)
        self.train_features = (self.train_features - mean_features) / std_features
        self.test_features = (self.test_features - mean_features) / std_features
        # concatenate bias
        self.train_features = torch.cat([torch.ones(self.train_num, 1, device = self.device), self.train_features], dim = 1)
        self.test_features = torch.cat([torch.ones(self.test_num, 1, device = self.device), self.test_features], dim = 1)

    def load_dataset_nn(self, dataset_name):
        main_folder = os.path.join('datasets')
        self.out_dim = 1
        # load and split dataset
        if dataset_name in 'YearPredictionMSD':  # no need to split dataset
            train_path = os.path.join(main_folder, dataset_name, dataset_name + '-train.txt')
            test_path = os.path.join(main_folder, dataset_name, dataset_name + '-test.txt')
            train_labels_raw, train_features_raw = svm_read_problem(train_path, return_scipy = True)
            test_labels_raw, test_features_raw = svm_read_problem(test_path, return_scipy = True)
            train_features_raw = train_features_raw.toarray()
            test_features_raw = test_features_raw.toarray()
        elif dataset_name in 'abalone, boston, mpg, cpusmall, cadata, space, mg':    # split dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.txt')
            labels_raw, features_raw = svm_read_problem(data_path, return_scipy = True)
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)
            train_features_raw = train_features_raw.toarray()
            test_features_raw = test_features_raw.toarray()
        elif dataset_name in 'concrete':    # load xls file and split the dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.xls')
            data_raw = pd.read_excel(data_path, header = 0).values
            labels_raw, features_raw = data_raw[:, -1], data_raw[:,:-1]
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)
        elif dataset_name in 'energy, kin8nm, casp, superconduct, slice, online, sgemm, electrical, churn':  # load csv file and split the dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.csv')
            data_raw = pd.read_csv(data_path, header = 0).values
            if dataset_name in 'energy':
                labels_raw, features_raw = data_raw[:, 1].astype(np.float32), data_raw[:, 2:].astype(np.float32)
            elif dataset_name in 'kin8nm, superconduct, slice':
                labels_raw, features_raw = data_raw[:, -1], data_raw[:, :-1]
            elif dataset_name in 'casp':
                labels_raw, features_raw = data_raw[:, 0], data_raw[:, 1:]
            elif dataset_name in 'online':
                labels_raw, features_raw = data_raw[:, -1].astype(np.float32), data_raw[:, 1:-1].astype(np.float32)
            elif dataset_name in 'sgemm':
                labels_raw, features_raw = data_raw[:, -4], data_raw[:, :-4]
            elif dataset_name in 'electrical':
                labels_raw, features_raw = data_raw[:, -2].astype(np.float32), data_raw[:, :-2].astype(np.float32)
            elif dataset_name in 'churn':
                labels_raw, features_raw = data_raw[:, -1].astype(np.float32), data_raw[:, 1:-1].astype(np.float32)
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)    
        elif dataset_name in 'WineRed, WineWhite':
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.csv')
            attris = pd.read_csv(data_path, header = 0).values.reshape(-1).tolist()
            data_raw = []
            for attr in attris:
                data_raw.append([eval(number) for number in attr.split(';')])
            data_raw = np.array(data_raw)
            labels_raw, features_raw = data_raw[:, -1], data_raw[:,:-1]
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)  
        else:   # TODO:  naval
            raise ValueError('dataset {} not found'.format(dataset_name))
        # to Tensor
        self.train_features = torch.from_numpy(train_features_raw).float().to(self.device)
        self.train_labels = torch.from_numpy(train_labels_raw).float().to(self.device)
        self.test_features = torch.from_numpy(test_features_raw).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels_raw).float().to(self.device)
        # Normalization
        self.std_features = torch.std(self.train_features, dim = 0, unbiased = True)
        self.std_features[self.std_features == 0] = 1
        self.mean_features = torch.mean(self.train_features, dim = 0)
        self.std_labels = torch.std(self.train_labels, dim = 0)
        self.mean_labels = torch.mean(self.train_labels, dim = 0)
        self.train_features = (self.train_features - self.mean_features) / self.std_features
        self.train_labels = (self.train_labels - self.mean_labels) / self.std_labels
        self.test_features = (self.test_features - self.mean_features) / self.std_features
        # self.test_labels = (self.test_labels - self.mean_labels) / self.std_labels
        # record information
        self.data_dim = len(self.train_features[0])
        self.train_num = len(self.train_features)
        self.test_num = len(self.test_features)
        self.model_dim = self.data_dim * self.n_hidden + self.n_hidden * self.out_dim +\
            self.n_hidden + self.out_dim + 2  # 2 variances
'''