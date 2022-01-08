import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, classification_out_dim):
        super(ResNetSimCLR, self).__init__()
        self.base_model = base_model
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(self.base_model)
        dim_mlp = self.backbone.fc.in_features


        # add MTL projection heads
        last_layer = self.backbone.fc
        self.backbone.fc = nn.Identity() # place holder for the fc layer

        # classification prediction head
        self.predict = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), 
                                              nn.Linear(dim_mlp, classification_out_dim, bias=True))

        # contrastive learning head
        self.contrast = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), last_layer)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        base_model = self.backbone(x)
        prediction_out = self.predict(base_model)
        simclr_out = self.contrast(base_model)
        return prediction_out, simclr_out
