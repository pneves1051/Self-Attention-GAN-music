import torch


class Reshape(nn.Module):
  def __init__(self, shape):
    super(Reshape, self).__init__()
    self.shape= shape
  
  def forward(self, x):
    return x.view(-1, *self.shape)

class SNConv1d(nn.Module):
  def __init__(self, *args, **kwargs):
    super(SNConv1d, self).__init__()
 
    self.conv = torch.nn.utils.spectral_norm(nn.Conv1d(*args, **kwargs))
 
  def forward(self, x):
    return self.conv(x)

class SNConvTranspose1d(nn.Module):
  def __init__(self, *args, **kwargs):
    super(SNConvTranspose1d, self).__init__()
 
    self.conv = torch.nn.utils.spectral_norm(nn.ConvTranspose1d(*args, **kwargs))
 
  def forward(self, x):
    return self.conv(x)

class SelfAttn(nn.Module):
  def __init__(self, ch, activation):
    super(SelfAttn, self).__init__()
    self.ch = ch
    self.activation = activation

    # Key
    self.theta = nn.Conv1d(self.ch, self.ch//8, 1, bias = False)
    self.phi = nn.Conv1d(self.ch, self.ch//8, 1, bias = False)
    self.g = nn.Conv1d(self.ch, self.ch//2, 1, bias=False)
    self.o = nn.Conv1d(self.ch//2, self.ch, 1, bias=False)
    
    # Gain parameter
    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

  def forward(self, x):

    # query
    theta = self.theta(x)
    # key
    phi = F.max_pool1d(self.phi(x), [2])
    # value
    g = F.max_pool1d(self.g(x), [2])

    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)))

    return self.gamma * o + x


class SNSelfAttn(nn.Module):
  def __init__(self, ch, activation):
    super(SNSelfAttn, self).__init__()
    self.ch = ch
    self.activation = activation

    # Key
    self.theta = nn.utils.spectral_norm(nn.Conv1d(self.ch, self.ch//8, 1, bias = False))
    self.phi = nn.utils.spectral_norm(nn.Conv1d(self.ch, self.ch//8, 1, bias = False))
    self.g = nn.utils.spectral_norm(nn.Conv1d(self.ch, self.ch//2, 1, bias=False))
    self.o = nn.utils.spectral_norm(nn.Conv1d(self.ch//2, self.ch, 1, bias=False))
    
    # Gain parameter
    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

  def forward(self, x):

    # query
    theta = self.theta(x)
    # key
    phi = F.max_pool1d(self.phi(x), [2])
    # value
    g = F.max_pool1d(self.g(x), [2])

    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)))

    return self.gamma * o + x


class ConditionalBatchNorm1d(nn.Module):
  def __init__(self, num_features, num_classes):
    super(ConditionalBatchNorm1d, self).__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm1d(num_features, affine=False)
    self.embed = nn.Embedding(num_classes, num_features * 2)
    self.embed.weight.data[:, :num_features].normal_(1, 0.02) # Initialize scale to 1
    self.embed.weight.data[:, num_features:].zero_() # Initialize bias at 0

  def forward(self, x, y):
    out = self.bn(x)
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1) * out + beta.view(-1, self.num_features, 1)
    return out

class ResLayer(nn.Module):
  def __init__(self, in_filters, out_filters, dilation, leaky=False, phase_shuffle=0):
    super(ResLayer, self).__init__()
    padding = dilation
    
    self.inp_conv = nn.Conv1d(in_filters, out_filters, 1)
 
    self.conv = [nn.Conv1d(in_filters, out_filters, 3, dilation=dilation, padding=padding)] +\
               ([nn.BatchNorm1d(out_filters)] if not leaky else []) +\
               ([nn.LeakyReLU(0.2)] if leaky else [nn.ReLU()])  +\
                [nn.Conv1d(out_filters, out_filters, 1)]  +\
               ([nn.BatchNorm1d(out_filters)] if not leaky else [])
 
       
    self.conv = nn.Sequential(*self.conv)
                  
    self.relu =  (nn.LeakyReLU(0.2)if leaky else nn.ReLU())
       
  def forward(self, inputs):
    res_x = self.conv(inputs)
    res_x += self.inp_conv(inputs)
    res_x = self.relu(res_x)
    return res_x

class ResBlock(nn.Module):
  def __init__(self, in_filters, out_filters, dilations, depth, leaky=False, phase_shuffle=2):
    super(ResBlock, self).__init__()
    self.res_block = nn.Sequential(*[ResLayer(in_filters if i==0 else out_filters, out_filters, dilations[i], leaky, phase_shuffle)
                                     for i in range(depth)])
 
  def forward(self, inputs):
    output = self.res_block(inputs)
    return output

class GenBlock(nn.Module):
  def __init__(self, in_filters, out_filters, num_classes, resample_scale):
    super(GenBlock, self).__init__()
    dilation = [1, 3, 9]
    padding = dilation
    
    #self.inp_resample = Resample(resample_scale)
    self.inp_resample = nn.Upsample(scale_factor=resample_scale)
    self.inp_conv = nn.Conv1d(in_filters, out_filters, 1)
     
    self.bn1 = ConditionalBatchNorm1d(in_filters, num_classes)
    self.relu1 = nn.ReLU()
    self.resample = nn.Upsample(scale_factor=resample_scale)
    #self.resample = Resample(resample_scale)
    self.conv1 = nn.Conv1d(in_filters, out_filters, 3, padding = dilation[0], dilation=dilation[0])

    self.bn2 = ConditionalBatchNorm1d(out_filters, num_classes)
    self.relu2 = nn.ReLU()
    self.conv2 = nn.Conv1d(out_filters, out_filters, 3, padding = dilation[1], dilation=dilation[1])

    self.bn3 = ConditionalBatchNorm1d(out_filters, num_classes)
    self.relu3 = nn.ReLU()
    self.conv3 = nn.Conv1d(out_filters, out_filters, 3, padding = dilation[2], dilation=dilation[2])
       
  def forward(self, x, labels):
    res_x = self.bn1(x, labels)
    res_x = self.relu1(res_x)
    res_x = self.resample(res_x)
    res_x = self.conv1(res_x)

    res_x = self.bn2(res_x, labels)
    res_x = self.relu2(res_x)
    res_x = self.conv2(res_x)

    res_x = self.bn3(res_x, labels)
    res_x = self.relu3(res_x)
    res_x = self.conv3(res_x)

    inp_x = self.inp_resample(x)
    inp_x = self.inp_conv(inp_x)

    res_x = inp_x + res_x
    return res_x

class DiscBlock(nn.Module):
  def __init__(self, in_filters, out_filters, down_scale):
    super(DiscBlock, self).__init__()
    dilation = [1, 3, 9]
    padding = dilation
    
    self.inp_pool = nn.AvgPool1d(down_scale)
    self.inp_conv = SNConv1d(in_filters, out_filters, 1)
     
    self.relu1 = nn.ReLU()
    self.pool = nn.AvgPool1d(down_scale)
    self.conv1 = SNConv1d(in_filters, out_filters, 3, padding = dilation[0], dilation=dilation[0])

    self.relu2 = nn.ReLU()
    self.conv2 = SNConv1d(out_filters, out_filters, 3, padding = dilation[1], dilation=dilation[1])

    self.relu3 = nn.ReLU()
    self.conv3 = SNConv1d(out_filters, out_filters, 3, padding = dilation[2], dilation=dilation[2])
       
  def forward(self, x):
    res_x = self.relu1(x)
    res_x = self.conv1(res_x)

    res_x = self.relu2(res_x)
    res_x = self.conv2(res_x)

    res_x = self.relu3(res_x)
    res_x = self.conv3(res_x)
    res_x = self.pool(res_x)

    inp_x = self.inp_conv(x)
    inp_x = self.inp_pool(inp_x)

    res_x = inp_x + res_x
    return res_x
