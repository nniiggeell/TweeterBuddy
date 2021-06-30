# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:31:54 2021

@author: agent
"""
import pandas as pd
import fake_news_model as fakenews

def input_func(text):
    lines2 = []
    lines2.append(text)
    df = pd.DataFrame({'text' : lines2}).astype(str)
    df['text'] = df['text'].apply(lambda x: x.lower())
    #print(df)
    # input_text = model.punctuation_removal(text)
    input_df = df.apply(fakenews.punctuation_removal)
    #print(input_df)
    prediction = fakenews.model.predict(input_df)
    return(str(prediction)[2:6])
text = "god speaks to me"
print(input_func("WASHINGTON (Reuters) - U.S. Senate Majority Leader Mitch McConnell on Friday said a shifting landscape will lead him to work with Democrats on immigration and financial regulation early in the new year, following a year of acrimony and partisan legislation. In an end-of-year news conference, McConnell touted a list of Republican accomplishments since President Donald Trump took office in January. It started with the confirmation of Neil Gorsuch to the Supreme Court and ended with an overhaul of the  U.S. tax code. But in January, McConnellâ€™s already razor-thin 52-48 Republican majority will shrink to 51-49 with the swearing in of Senator-elect Doug Jones, the Democrat who surprised the political world with a win in a special election in the deeply Republican state of Alabama. Adding to McConnellâ€™s difficulties, special Senate procedures are fading that allowed him to pass a tax bill and try to repeal the Affordable Care Act this year without any Democratic support. That means that McConnellâ€™s victories - if he has them - will require more collaboration and less confrontation. The pivot was the centerpiece of his news conference remarks.  â€œThere are areas where I think we can get bipartisan agreement,â€ McConnell said. First on his list was legislation to change Dodd-Frank banking regulations that he said would help smaller financial institutions. The Kentucky senator noted that Senate Banking Committee Chairman Mike Crapo has advanced legislation that is co-sponsored by several Democrats. McConnell also pointed to bipartisan efforts to help undocumented immigrants, known as â€œDreamers,â€ who were brought into the United States when they were children. If negotiators from both parties can come to a deal for the Dreamers that Trumpâ€™s administration can support, â€œweâ€™ll spend floor time on that in January,â€ McConnell said. On Thursday, Senate Democratic Leader Chuck Schumer complained that throughout 2017 Republicans â€œhave been hell-bent on pursuing a partisan agenda.â€ When asked by a reporter of possible bipartisan successes in 2018, Schumer pointed to the need for infrastructure improvements but said that Trump has been â€œall over the lotâ€ on how to accomplish road, airport and other construction projects. With the November 2018 congressional elections approaching, Democrats might have less incentive to cooperate with Republicans, especially after Schumerâ€™s party won decisive victories in special elections this month and last in Alabama and Virginia. McConnell hinted it would be tougher to find agreement with Democrats on some other legislative issues, including welfare reform, which Trump says he wants to push ahead with in 2018. McConnell said he would consult with Trump and House of Representatives Speaker Paul Ryan in January over prospects for welfare reform."  ))