import pandas as pd

class DataMatcher:
    def __init__(self, df, df_mc):
        self.df = df
        self.df_mc = df_mc

    def get_mc(self, event, trackMcID):
        return self.df_mc.query(f"event == {event} and id == {trackMcID}").iloc[0]
    
    def add_mc_info(self, cols):
        merged = self.df.merge(
            self.df_mc[["event", "id"] + cols], 
            left_on=["event", "mcTrackID"], 
            right_on=["event", "id"],
            how="left"
        )

        merged = merged.drop(columns=["id"])

        return merged