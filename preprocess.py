import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


class Converter:
    def __init__(self):
        self.pitch_columns = {
                'データ内連番': 'num_data',
				'球種': 'ball_type',  # 目的変数
				'投球位置区域': 'corse',  # 目的変数
				'年度': '',
				'試合ID': '',
				'試合内連番': '',
				'試合内投球数': 'num_game_throw',
				'日付': '',
				'時刻': '',
                'ホームチームID': '',
				'アウェイチームID': '',
				'球場ID': '',
				'球場名': '',
				'試合種別詳細': '',
				'イニング': 'ord_inning',
				'表裏': '',
                'イニング内打席数': 'num_inning_bat',
				'打席内投球数': 'num_pitch_in_bat',
				'投手ID': '',
				'投手チームID': '',
				'投手投球左右': 'pitcher_lr',
				'投手役割': 'pitcher_role',
				'投手登板順': '',
                '投手試合内対戦打者数': '',
				'投手試合内投球数': '',
				'投手イニング内投球数': '',
				'打者ID': '',
				'打者チームID': '',
				'打者打席左右': 'bat_lr',
                '打者打順': 'ord_bat',
				'打者守備位置': '',
				'打者試合内打席数': '',
				'プレイ前ホームチーム得点数': '',
				'プレイ前アウェイチーム得点数': '',
                'プレイ前アウト数': 'num_out',
				'プレイ前ボール数': 'num_ball',
				'プレイ前ストライク数': 'num_strike',
				'プレイ前走者状況': 'runner_state',
				'一塁走者ID': '',
				'二塁走者ID': '',
                '三塁走者ID': '',
				'捕手ID': '',
				'一塁手ID': '',
				'二塁手ID': '',
				'三塁手ID': '',
				'遊撃手ID': '',
				'左翼手ID': '',
				'中堅手ID': '',
                '右翼手ID': '',
				'成績対象投手ID': '',
				'成績対象打者ID': '',
            }
        self.player_columns = {

        }

    def change_to_cat(self, df):
        for jp_clmn in ["投手投球左右", "投手投球左右", "投手役割", "打者打席左右", "プレイ前走者状況"]:
            clmn = self.pitch_columns[jp_clmn]
            le_ = le.fit(df[clmn])
            df["cat_"+clmn] = le_.transform(df[clmn])

        return df


    def preprocess_pitch(self, df_, isTrain):
        df = df_.copy()
        if isTrain:
            df = df.rename(columns=self.pitch_columns)
        else:
            tmp_pitch_columns = self.pitch_columns.copy()
            tmp_pitch_columns.pop("球種")
            df = df.rename(columns=tmp_pitch_columns)
        df = self.change_to_cat(df)

        return df


    def preprocess_player(self, df_):
        df = df_.copy()
        df = df.rename(columns=self.player_columns)
        return df

    def convert_df(self, df_pitch_, df_player_, isTrain=True):
        df_pitch = self.preprocess_pitch(df_pitch_, isTrain)
        df_player = self.preprocess_player(df_player_)

        # TODO: merge処理
        return df_pitch
