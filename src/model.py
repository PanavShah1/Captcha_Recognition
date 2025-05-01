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
    
# class DeepCaptchaModel(nn.Module):
#     def __init__(self, num_chars, img_height=75, img_width=300):
#         super(DeepCaptchaModel, self).__init__()

#         # Convolutional Backbone
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         self.conv_3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 1))  # only pool height a bit here to keep width longer for GRU
#         )

#         self.conv_4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 1))
#         )

#         self.fc = nn.Linear(2048, 256)

#         # GRU for sequence modeling
#         self.gru = nn.GRU(
#             input_size=256,
#             hidden_size=128,
#             num_layers=2,
#             # batch_first=True,
#             bidirectional=True,
#             dropout=0.25
#         )

#         # Output
#         self.output = nn.Linear(128 * 2, num_chars + 1)  # +1 for CTC blank token

#     def _forward_conv(self, x):
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         x = self.conv_3(x)
#         x = self.conv_4(x)
#         # print(x.shape)
#         return x

#     def forward(self, images, targets=None):
#         bs, c, h, w = images.size()

#         x = self._forward_conv(images)
#         # print(x.shape)
#         # Prepare for RNN
#         x = x.permute(0, 3, 1, 2)  # [batch, width, channels, height]
#         # print(x.shape)
#         x = x.view(bs, x.size(1), -1)  # Flatten channels and height
#         # print(x.shape)
#         x = self.fc(x)
#         # print(x.shape)

#         # GRU
#         x, _ = self.gru(x)

#         # Final output
#         x = self.output(x)
#         # print(x.shape)

#         # CTC loss expects (T, N, C)
#         x = x.permute(1, 0, 2)
#         # print(x.shape)

#         if targets is not None:
#             log_softmax_values = F.log_softmax(x, dim=2)
#             input_lengths = torch.full(size=(bs,), fill_value=log_softmax_values.size(0), dtype=torch.int32)
#             target_lengths = torch.full(size=(bs,), fill_value=targets.size(1), dtype=torch.int32)
#             loss = nn.CTCLoss(blank=0)(log_softmax_values, targets, input_lengths, target_lengths)
#             return x, loss

#         return x, None
    
class DeepCaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(DeepCaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=2)  # added stride=2
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2)  # added stride=2
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.linear_1 = nn.Linear(1216, 64)  # 9 comes from the reduced height, see below
        self.drop_1 = nn.Dropout(0.2)

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, c, h, w = images.size()

        x = F.relu(self.bn1(self.conv_1(images)))  # [bs, 128, h/2, w/2]
        x = F.relu(self.bn2(self.conv_2(x)))       # [bs, 64, h/4, w/4]
        x = F.relu(self.bn3(self.conv_3(x)))        # [bs, 64, h/4, w/4]

        x = x.permute(0, 3, 1, 2)  # [bs, width, channels, height]
        x = x.view(bs, x.size(1), -1)  # flatten channels and height

        x = self.linear_1(x)
        x = self.drop_1(x)
        x, _ = self.gru(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)

        if targets is not None:
            log_softmax_values = F.log_softmax(x, dim=2)
            input_lengths = torch.full(
                size=(bs,),
                fill_value=log_softmax_values.size(0),
                dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,),
                fill_value=targets.size(1),
                dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, target_lengths
            )
            return x, loss
        return x, None
    
class DeepCaptchaModelSmallerTimeSteps(nn.Module):
    def __init__(self, num_chars):
        super(DeepCaptchaModelSmallerTimeSteps, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=2)  # added stride=2
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2)  # added stride=2
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=(1, 2))  # added stride=2
        self.bn3 = nn.BatchNorm2d(64)

        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=(1, 2))  # added stride=2
        self.bn4 = nn.BatchNorm2d(64)

        self.linear_1 = nn.Linear(1216, 256)  # 9 comes from the reduced height, see below
        self.drop_1 = nn.Dropout(0.2)

        self.linear_2 = nn.Linear(256, 64)
        self.drop_2 = nn.Dropout(0.2)

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, c, h, w = images.size() # [bs, 3, 75, 300]
        # print(images.size())
        x = F.relu(self.bn1(self.conv_1(images)))  # [bs, 128, 38, 150]
        # print(x.size())
        x = F.relu(self.bn2(self.conv_2(x)))       # [bs, 64, 19, 75]
        # print(x.size())
        x = F.relu(self.bn3(self.conv_3(x)))        # [bs, 64, 19, 37]
        # print(x.size())
        x = F.relu(self.bn4(self.conv_4(x)))        # [bs, 64, 19, 19]
        # print(x.size())

        x = x.permute(0, 3, 1, 2)  # [bs, width, channels, height] = [bs, 19, 64, 19]
        # print(x.size())
        x = x.view(bs, x.size(1), -1)  # flatten channels and height = [bs, 19, 1216]
        # print(x.size())

        x = F.relu(self.linear_1(x)) # [bs, 19, 256]
        # print(x.size())
        x = self.drop_1(x) 
        x = F.relu(self.linear_2(x)) # [bs, 19, 64]
        # print(x.size())
        x = self.drop_2(x)

        x, _ = self.gru(x)
        # print(x.size())
        x = self.output(x)
        # print(x.size())
        x = x.permute(1, 0, 2)
        # print(x.size())

        if targets is not None:
            log_softmax_values = F.log_softmax(x, dim=2)
            input_lengths = torch.full(
                size=(bs,),
                fill_value=log_softmax_values.size(0),
                dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,),
                fill_value=targets.size(1),
                dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, target_lengths
            )
            return x, loss
        return x, None


if __name__ == "__main__":
    cm = DeepCaptchaModel(num_chars=19)
    img = torch.rand(5, 3, 75, 300)
    target = torch.randint(1, 20, (5, 5))
    x, loss = cm(img, target)

