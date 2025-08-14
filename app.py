def calculate_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    df['td_setup'] = 0
    df['td_countdown'] = 0
    df['td_nine'] = False
    df['td_thirteen'] = False

    for i in range(4, n):
        if df['close'].iloc[i] < df['close'].iloc[i-4]:
            if df['td_setup'].iloc[i-1] > 0:
                df.loc[i, 'td_setup'] = 1
            else:
                df.loc[i, 'td_setup'] = df['td_setup'].iloc[i-1] - 1
        elif df['close'].iloc[i] > df['close'].iloc[i-4]:
            if df['td_setup'].iloc[i-1] < 0:
                df.loc[i, 'td_setup'] = 1
            else:
                df.loc[i, 'td_setup'] = df['td_setup'].iloc[i-1] + 1
        else:
            df.loc[i, 'td_setup'] = 0

    df['td_nine'] = (df['td_setup'].abs() == 9)

    countdown_active = False
    countdown_count = 0
    countdown_direction = 0

    for i in range(2, n):
        if df['td_nine'].iloc[i]:
            countdown_active = True
            countdown_count = 0
            countdown_direction = 1 if df['td_setup'].iloc[i] < 0 else -1

        if countdown_active and i >= 2:
            if countdown_direction == 1 and df['close'].iloc[i] <= df['low'].iloc[i-2]:
                countdown_count += 1
            elif countdown_direction == -1 and df['close'].iloc[i] >= df['high'].iloc[i-2]:
                countdown_count += 1
            df.loc[i, 'td_countdown'] = countdown_count
            if countdown_count >= 13:
                df.loc[i, 'td_thirteen'] = True
                countdown_active = False

    return df
