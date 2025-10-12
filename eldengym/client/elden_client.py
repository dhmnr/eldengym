from .siphon_client import SiphonClient
import numpy as np
from time import sleep

class EldenClient(SiphonClient):
    """
    Client for the Elden Ring game.
    """
    def __init__(self, host='localhost:50051', **kwargs):
        super().__init__(host, **kwargs)
        self.scenarios = {
            'margit': {
                'boss_name': 'Margit',
                'fog_wall_location': (19.958229064941406, -11.990748405456543, -7.051832675933838),
            }
        }
    ## =========== Player methods ===========
    @property
    def player_hp(self):
        """
        Get the health of the player.
        """
        return self.get_attribute('HeroHp')
    
    @property
    def player_max_hp(self):
        """
        Get the maximum health of the player.
        """
        return self.get_attribute('HeroMaxHp')

    def set_player_hp(self, hp):
        """
        Set the health of the player.
        """
        self.set_attribute('HeroHp', hp)

    @property
    def local_player_coords(self):
        """
        Get the location of the player.
        """
        local_x = self.get_attribute('HeroLocalPosX')
        local_y = self.get_attribute('HeroLocalPosY')
        local_z = self.get_attribute('HeroLocalPosZ')
        return local_x, local_y, local_z

    @property
    def global_player_coords(self):
        """
        Get the location of the player.
        """
        global_x = self.get_attribute('HeroGlobalPosX')
        global_y = self.get_attribute('HeroGlobalPosY')
        global_z = self.get_attribute('HeroGlobalPosZ')
        return global_x, global_y, global_z

    @property
    def player_animation_id(self):
        """
        Get the animation id of the player.
        """
        return self.get_attribute('HeroAnimId')

    ## =========== Target methods ===========
    @property
    def target_hp(self):
        """
        Get the health of the target.
        """
        return self.get_attribute('NpcHp')
    
    @property
    def target_max_hp(self):
        """
        Get the maximum health of the target.
        """
        return self.get_attribute('NpcMaxHp')
    
    def set_target_hp(self, hp):
        """
        Set the health of the target.
        """
        self.set_attribute('NpcHp', hp)
    
    @property
    def local_target_coords(self):
        """
        Get the location of the target.
        """
        local_x = self.get_attribute('NpcLocalPosX')
        local_y = self.get_attribute('NpcLocalPosY')
        local_z = self.get_attribute('NpcLocalPosZ')
        return local_x, local_y, local_z
    
    @property
    def global_target_coords(self):
        """
        Get the location of the target.
        """
        global_x = self.get_attribute('NpcGlobalPosX')
        global_y = self.get_attribute('NpcGlobalPosY')
        global_z = self.get_attribute('NpcGlobalPosZ')
        return global_x, global_y, global_z
    
    @property
    def target_animation_id(self):
        """
        Get the animation id of the target.
        """
        return self.get_attribute('NpcAnimId')

    
    ## =========== Helper methods ===========
    @property
    def target_player_distance(self):
        """
        Get the distance between the player and the target.
        """
        player_x, player_y, player_z = self.local_player_coords
        target_x, target_y, target_z = self.global_target_coords
        return np.linalg.norm([player_x - target_x, player_y - target_y, player_z - target_z])

    def teleport(self, x, y, z):
        """
        Teleport the player to the given coordinates.
        """
        # FIXME: Close range teleport, need to check MapId for long range teleport.
        local_x, local_y, local_z = self.local_player_coords
        global_x, global_y, global_z = self.global_player_coords
        self.set_attribute('HeroLocalPosX', local_x + (x - global_x))
        self.set_attribute('HeroLocalPosY', local_y + (y - global_y))
        self.set_attribute('HeroLocalPosZ', local_z + (z - global_z))

    def set_game_speed(self, speed):
        """
        Set the game speed.
        """
        self.set_attribute('gameSpeedFlag', True)
        self.set_attribute('gameSpeedVal', speed)

    def reset_game(self):
        """
        Reset the game by setting the player's hp to 0.
        """
        self.set_player_hp(0)
        sleep(20) # FIXME: This is a hack to wait for the game to reset, doesn't work well.

    def start_scenario(self, scenario_name='Margit'):
        """
        Start the scenario with the given scenario name.
        """
        # FIXME: This is a hack to start boss fight. Need to check fogwall state. or use another method.
        x,y,z = self.scenarios[scenario_name]['fog_wall_location']
        self.teleport(x, y, z)
        self.move_mouse(1000, 0, 1)
        sleep(2)
        self.send_key(['W','E'], 200, 200)
        sleep(2)
        self.send_key(['B'], 200)



        