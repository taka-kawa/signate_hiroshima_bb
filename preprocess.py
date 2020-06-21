import copy
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

class Converter:
    def __init__(self):
        self.pitch_columns = {
            'データ内連番': 'num_data',
            '球種': 'ball_type',  # 目的変数
            '投球位置区域': 'corse',  # 目的変数
            '年度': 'year',
            '試合ID': '',
            '試合内連番': '',
            '試合内投球数': 'num_game_throw',
            '日付': '',
            '時刻': '',
            'ホームチームID': 'home_id',
            'アウェイチームID': 'away_id',
            '球場ID': 'dome_id',
            '球場名': '',
            '試合種別詳細': 'game_detail',
            'イニング': 'ord_inning',
            '表裏': 'fb',
            'イニング内打席数': 'num_inning_bat',
            '打席内投球数': 'num_pitch_in_bat',
            '投手ID': 'pitcher_id',
            '投手チームID': 'pitcher_team_id',
            '投手投球左右': 'pitcher_lr',
            '投手役割': 'pitcher_role',
            '投手登板順': 'ord_pitching',
            '投手試合内対戦打者数': 'num_pitcher_vs',
            '投手試合内投球数': 'num_pitching_game',
            '投手イニング内投球数': 'num_pitching_inning',
            '打者ID': 'batter_id',
            '打者チームID': 'batter_team_id',
            '打者打席左右': 'bat_lr',
            '打者打順': 'ord_bat',
            '打者守備位置': 'batter_def_posi',
            '打者試合内打席数': 'num_batter_batting',
            'プレイ前ホームチーム得点数': 'num_point_home',
            'プレイ前アウェイチーム得点数': 'num_point_away',
            'プレイ前アウト数': 'num_out',
            'プレイ前ボール数': 'num_ball',
            'プレイ前ストライク数': 'num_strike',
            'プレイ前走者状況': 'runner_state',
            '一塁走者ID': '',
            '二塁走者ID': '',
            '三塁走者ID': '',
            '捕手ID': 'catcher_id',
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
            '年度': 'year',
            'チームID': '',
            'チーム名': '',
            '選手ID': 'player_id',
            '選手名': '',
            '育成選手F': 'training_player',
            '背番号': '',
            '位置': 'position',
            '投': '',
            '打': '',
            '身長': 'num_height',
            '体重': 'num_weight',
            '生年月日': '',
            '出身高校ID': '',
            '出身高校名': '',
            '出身大学ID': '',
            '出身大学名': '',
            '社会人': '',
            'ドラフト年': '',
            'ドラフト種別': '',
            'ドラフト順位': 'ord_draft',
            '年俸': 'num_annual_salary',
            '出身国': 'country',
            '出身地': '',
            '血液型': '',
        }
        self.le = LabelEncoder()
        self.les = {}
        self.unknown = 99999999

    def change_to_cat(self, df, jp_clmns, d_type, isTrain):
        for jp_clmn in jp_clmns:
            if d_type == "pitch":
                clmn = self.pitch_columns[jp_clmn]
            elif d_type == "player":
                clmn = self.player_columns[jp_clmn]

            if isTrain:
                if df[clmn].dtype == 'O':
                    self.le_ = self.le.fit(pd.concat([df[clmn], pd.Series(["self.unknown"])]))  # 未知ラベル用のラベルを追加
                else:
                    self.le_ = self.le.fit(pd.concat([df[clmn], pd.Series([self.unknown])]))  # 未知ラベル用のラベルを追加
                self.les[jp_clmn] = copy.copy(self.le_)
            
            # 未知ラベルの処理
            if not isTrain:
                df[clmn] = df[clmn].map(lambda x: self.unknown if x not in self.les[jp_clmn].classes_ else x)
                self.les[jp_clmn].classes_ = np.append(self.les[jp_clmn].classes_, self.unknown)

            df["cat_"+clmn] = self.les[jp_clmn].transform(df[clmn])
        
        return df

    def preprocess_pitch(self, df_, isTrain):
        df = df_.copy()
        if isTrain:
            df = df.rename(columns=self.pitch_columns)
        else:
            tmp_pitch_columns = self.pitch_columns.copy()
            tmp_pitch_columns.pop("球種")
            df = df.rename(columns=tmp_pitch_columns)

        # カテゴリ変換
        jp_clmns = [
            "投手投球左右",
            "投手役割",
            "打者打席左右",
            "プレイ前走者状況",
            "ホームチームID",
            "アウェイチームID",
            "球場ID",
            "試合種別詳細",
            "表裏",
            "投手チームID",
            "打者チームID",
            "打者守備位置",
            "捕手ID",
            ]
        df = self.change_to_cat(df, jp_clmns, "pitch", isTrain)

        return df


    def preprocess_player(self, df_, isTrain):
        df = df_.copy()
        df = df.rename(columns=self.player_columns)

        # カテゴリ変換
        jp_clmns = ["育成選手F", "位置", "出身国"]
        df = self.change_to_cat(df, jp_clmns, "player", isTrain)
        return df

    def convert_df(self, df_pitch_, df_player_, isTrain=True):
        df_pitch = self.preprocess_pitch(df_pitch_, isTrain)
        df_player = self.preprocess_player(df_player_, isTrain)

        suffix = "_batter"
        final_df = pd.merge(df_pitch, df_player.add_suffix(suffix), left_on=[self.pitch_columns["年度"], self.pitch_columns["打者ID"]], right_on=[self.player_columns["年度"]+suffix, self.player_columns["選手ID"]+suffix], how="left")
        suffix = "_pitcher"
        final_df = pd.merge(final_df, df_player.add_suffix(suffix), left_on=[self.pitch_columns["年度"], self.pitch_columns["投手ID"]], right_on=[self.player_columns["年度"]+suffix, self.player_columns["選手ID"]+suffix], how="left")
        return final_df