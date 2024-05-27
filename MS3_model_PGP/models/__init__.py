from motionnet.models.ptr.ptr import PTR
from motionnet.models.pgp.pgp import myPGP


from motionnet.models.pgp.encoders.encoder import PredictionEncoder
from motionnet.models.pgp.aggregators.aggregator import PredictionAggregator
from motionnet.models.pgp.decoders.decoder import PredictionDecoder


from models.pgp.encoders.raster_encoder import RasterEncoder
from models.pgp.encoders.polyline_subgraph import PolylineSubgraphs
from models.pgp.encoders.pgp_encoder import PGPEncoder
from models.pgp.aggregators.concat import Concat
from models.pgp.aggregators.global_attention import GlobalAttention
from models.pgp.aggregators.goal_conditioned import GoalConditioned
from models.pgp.aggregators.pgp import PGP
from models.pgp.decoders.mtp import MTP
from models.pgp.decoders.multipath import Multipath
from models.pgp.decoders.covernet import CoverNet
from models.pgp.decoders.lvm import LVM



from typing import List, Dict, Union


def initialize_pgp(config, encoder_type: str, aggregator_type: str, decoder_type: str,
                                encoder_args: Dict, aggregator_args: Union[Dict, None], decoder_args: Dict):
    """
    Helper function to initialize appropriate encoder, aggegator and decoder models
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = myPGP(config, encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        'raster_encoder': RasterEncoder,
        'polyline_subgraphs': PolylineSubgraphs,
        'pgp_encoder': PGPEncoder
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        'concat': Concat,
        'global_attention': GlobalAttention,
        'gc': GoalConditioned,
        'pgp': PGP
    }

    if aggregator_args:
        return aggregator_mapping[aggregator_type](aggregator_args)
    else:
        return aggregator_mapping[aggregator_type]()


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        'mtp': MTP,
        'multipath': Multipath,
        'covernet': CoverNet,
        'lvm': LVM
    }

    return decoder_mapping[decoder_type](decoder_args)

__all__ = {
    'ptr': PTR,
    'pgp': initialize_pgp,
}



def build_model(config):
    if config.method.model_name == 'pgp':
       model = __all__[config.method.model_name](config,
            config['encoder_type'], config['aggregator_type'], config['decoder_type'],
                                                config['encoder_args'], config['aggregator_args'], config['decoder_args'])
       
    else:
        model = __all__[config.method.model_name](
            config=config
        )
        


    return model


################################

# from motionnet.models.ptr.ptr import PTR
# from motionnet.models.pgp.pgp import myPGP

# import models.pgp.encoders.encoder as enc
# import models.pgp.aggregators.aggregator as agg
# import models.pgp.decoders.decoder as dec


# __all__ = {
#     'ptr': PTR,
#     'pgp': myPGP,
# }


# def build_model(config):
#     if config.method.model_name == 'ptr':
#         model = __all__[config.method.model_name](
#         config=config,
#     )
#     else: 
#         model = __all__[config.method.model_name](
#         config=config,
#         encoder = enc,
#         aggregator = agg,
#         decoder = dec
#     )

#     return model
