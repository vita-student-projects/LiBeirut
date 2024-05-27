import torch
import torch.nn as nn
import models.pgp.encoders.encoder as enc
import models.pgp.aggregators.aggregator as agg
import models.pgp.decoders.decoder as dec
from typing import Dict, Union
from motionnet.models.base_model.base_model import BaseModel

from typing import List, Dict, Union

class myPGP(BaseModel):
    """
    Single-agent prediction model
    """
    def __init__(self, config, encoder: enc.PredictionEncoder,
                 aggregator: agg.PredictionAggregator,
                 decoder: dec.PredictionDecoder):
        """
        Initializes model for single-agent trajectory prediction
        """
        print("iinittttttttttttttttttttt")

        super(myPGP, self).__init__(config)
        self.encoder = encoder
        self.aggregator = aggregator
        self.decoder = decoder

    # def forward(self, inputs: Dict) -> Union[torch.Tensor, Dict]:
    def forward(self, batch, batch_idx):
        """
        Forward pass for prediction model
        :param inputs: Dictionary with
            'target_agent_representation': target agent history
            'surrounding_agent_representation': surrounding agent history
            'map_representation': HD map representation
        :return outputs: K Predicted trajectories and/or their probabilities
        """
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'],inputs['obj_trajs_mask'] ,inputs['map_polylines']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1,1,1,1).repeat(1,1,*agents_in.shape[-2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1,1,1).repeat(1,1,agents_mask.shape[-1])).squeeze(1)
        agents_in = torch.cat([agents_in[...,:2],agents_mask.unsqueeze(-1)],dim=-1)
        agents_in = agents_in.transpose(1,2)
        ego_in = torch.cat([ego_in[...,:2],ego_mask.unsqueeze(-1)],dim=-1)
        roads = torch.cat([inputs['map_polylines'][...,:2],inputs['map_polylines_mask'].unsqueeze(-1)],dim=-1)
        
        model_input = {
            'target_agent_representation': ego_in,
            'surrounding_agent_representation': {
                'vehicles': agents_in,
                'vehicle_masks': agents_mask,
                'pedestrians': torch.zeros_like(agents_in),  # Assuming no pedestrian data
                'pedestrian_masks': torch.zeros_like(agents_mask)  # Assuming no pedestrian mask data
            },
            'map_representation': {
                'lane_node_feats': roads,
                'lane_node_masks': inputs['map_polylines_mask']
            },
            'agent_node_masks': {
                'vehicles': torch.ones_like(agents_mask),  # Adjust based on your masking strategy
                'pedestrians': torch.zeros_like(agents_mask)  # Assuming no pedestrian mask data
            }
        }

        if 's_next' in inputs and 'edge_type' in inputs:
            model_input['map_representation']['s_next'] = inputs['s_next']
            model_input['map_representation']['edge_type'] = inputs['edge_type']

        encodings = self.encoder(model_input)
        agg_encoding = self.aggregator(encodings)
        outputs = self.decoder(agg_encoding)

        return outputs
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['optim_args']['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config['optim_args']['scheduler_step'],
                                                            gamma=self.config['optim_args']['scheduler_gamma'])
        return [optimizer], [scheduler]
    