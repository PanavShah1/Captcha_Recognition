import torch
from torch import nn
from torch.nn import functional as F

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        self.output = nn.Linear(64, num_chars + 1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)



    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        # print(bs, c, h, w)
        x = F.relu(self.bn1(self.conv_1(images)))
        # print(x.size())
        x = self.max_pool_1(x)
        # print(x.size())
        x = F.relu(self.bn2(self.conv_2(x)))
        # print(x.size())
        x = self.max_pool_2(x) # [1, 64, 18, 75]
        # print(x.size())
        x = x.permute(0, 3, 1, 2) # [1, 75, 64, 18]
        # print(x.size())
        x = x.view(bs, x.size()[1], -1)
        # print(x.size())
        x = self.linear_1(x)
        x = self.drop_1(x)
        # print(x.size())
        x, _ = self.gru(x)
        # print(x.size())
        x = self.output(x)
        # print(x.size())
        x = x.permute(1, 0, 2) 
        # print(x.size())
        if targets is not None:
            log_softmax_values = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs, ), 
                fill_value=log_softmax_values.size(0),
                dtype=torch.int32
            )
            # print(input_lengths)
            target_lengths = torch.full(
                size=(bs, ), 
                fill_value=targets.size(1),
                dtype=torch.int32
            )
            # print(target_lengths)
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, target_lengths
            )
            return x, loss
        return x, None
    
class DeepCaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(DeepCaptchaModel, self).__init__()
        
        # Convolutional layers (with increased depth)
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_3 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Adding a residual connection to the third convolutional layer
        self.conv_res = nn.Conv2d(128, 512, kernel_size=(1, 1), padding=(0, 0))  # Residual mapping

        # Linear layers (for added complexity)
        self.linear_1 = nn.Linear(512 * 9 * 37, 128)  # Fix this dimension
        self.linear_2 = nn.Linear(128, 64)
        self.drop_1 = nn.Dropout(0.3)

        # Increased GRU layer complexity
        self.gru = nn.GRU(64, 64, bidirectional=True, num_layers=3, dropout=0.3)
        self.output = nn.Linear(128, num_chars + 1)

    def forward(self, images, targets=None):
        bs, c, h, w = images.size()

        # Convolutional pass
        x = F.relu(self.bn1(self.conv_1(images)))
        x = self.max_pool_1(x)
        
        x = F.relu(self.bn2(self.conv_2(x)))
        x = self.max_pool_2(x)
        
        x = F.relu(self.bn3(self.conv_3(x)))
        x = self.max_pool_3(x)
        
        # Apply residual connection (skip connection)
        x_res = self.conv_res(self.max_pool_1(F.relu(self.bn1(self.conv_1(images)))))
        x = x + x_res  # Element-wise addition (residual skip connection)
        
        # Reshape for GRU
        x = x.permute(0, 3, 1, 2)  # [batch_size, width, 512, height]
        x = x.view(bs, x.size(1), -1)  # Flatten height
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        x = F.relu(self.linear_2(x))

        # GRU pass (with more layers and bidirectional)
        x, _ = self.gru(x)
        
        # Final output layer
        x = self.output(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, num_classes]
        
        if targets is not None:
            log_softmax_values = F.log_softmax(x, 2)
            input_lengths = torch.full((bs,), w // 2, dtype=torch.int32)  # Adjust according to width
            target_lengths = torch.sum(targets != 0, dim=1)  # Target lengths
            
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, target_lengths
            )
            return x, loss
        
        return x, None


if __name__ == "__main__":
    cm = CaptchaModel(num_chars=19)
    img = torch.rand(5, 3, 75, 300)
    target = torch.randint(1, 20, (5, 5))
    x, loss = cm(img, target)

