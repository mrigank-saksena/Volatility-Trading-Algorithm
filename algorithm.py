import cvxopt
from functools import partial
from quantopian.pipeline import Pipeline  
from quantopian.algorithm import attach_pipeline, pipeline_output 
from quantopian.pipeline.data.builtin import USEquityPricing  
import math
import numpy as np
import scipy
from quantopian.pipeline.data.quandl import cboe_vix as vix
import zipline
from scipy import stats
import statsmodels as sm
from statsmodels.stats.stattools import jarque_bera

 
def initialize(context): 
    
    stock_ids = [symbol('AAPL'), symbol('ABT'), symbol('ACN'), symbol('ACE'), symbol('ADBE'), symbol('ADT'), symbol('AAP'), symbol('AES'), symbol('AET'), symbol('AFL'), symbol(
'AMG'), symbol('A'), symbol('GAS'), symbol('ARE'), symbol('APD'), symbol('AKAM'), symbol('AA'), symbol('AGN'), symbol('ALXN'), symbol('ALLE'), symbol('ADS'), symbol('ALL'), symbol('ALTR'), symbol('MO'), symbol('AMZN'), symbol('AEE'), symbol('AAL'), symbol('AEP'), symbol('AXP'), symbol('AIG'), symbol('AMT'), symbol('AMP'), symbol('ABC'), symbol('AME'), symbol('AMGN'), symbol('APH'), symbol('APC'), symbol('ADI'), symbol('AON'), symbol('APA'), symbol('AIV'), symbol('AMAT'), symbol('ADM'), symbol('AIZ'), symbol('T'), symbol('ADSK'), symbol('ADP'), symbol('AN'), symbol('AZO'), symbol('AVGO'), symbol('AVB'), symbol('AVY'), symbol('BHI'), symbol('BLL'), symbol('BAC'), symbol('BK'), symbol('BCR'), symbol('BAX'), symbol('BBT'), symbol('BDX'), symbol('BBBY'), symbol('BRK_B'), symbol('BBY'), symbol('BLX'), symbol('HRB'), symbol('BA'), symbol('BWA'), symbol('BXP'), symbol('BSX'), symbol('BMY'), symbol('BRCM'), symbol('BF.B'), symbol('CHRW'), symbol('CA'), symbol('CVC'), symbol('COG'), symbol('CAM'), symbol('CPB'), symbol('COF'), symbol('CAH'), symbol('HSIC'), symbol('KMX'), symbol('CCL'), symbol('CAT'), symbol('CBG'), symbol('CBS'), symbol('CELG'), symbol('CNP'), symbol('CTL'), symbol('CERN'), symbol('CF'), symbol('SCHW'), symbol('CHK'), symbol('CVX'), symbol('CMG'), symbol('CB'), symbol('CI'), symbol('XEC'), symbol('CINF'), symbol('CTAS'), symbol('CSCO'), symbol('C'), symbol('CTXS'), symbol('CLX'), symbol('CME'), symbol('CMS'), symbol('COH'), symbol('KO'), symbol('CCE'), symbol('CTSH'), symbol('CL'), symbol('CMCSA'), symbol('CMA'), symbol('CSC'), symbol('CAG'), symbol('COP'), symbol('CNX'), symbol('ED'), symbol('STZ'), symbol('GLW'), symbol('COST'), symbol('CCI'), symbol('CSX'), symbol('CMI'), symbol('CVS'), symbol('DHI'), symbol('DHR'), symbol('DRI'), symbol('DVA'), symbol('DE'), symbol('DLPH'), symbol('DAL'), symbol('XRAY'), symbol('DVN'), symbol('DO'), symbol('DTV'), symbol('DFS'), symbol('DISCA'), symbol('DISCK'), symbol('DG'), symbol('DLTR'), symbol('D'), symbol('DOV'), symbol('DOW'), symbol('DPS'), symbol('DTE'), symbol('DD'), symbol('DUK'), symbol('DNB'), symbol('ETFC'), symbol('EMN'), symbol('ETN'), symbol('EBAY'), symbol('ECL'), symbol('EIX'), symbol('EW'), symbol('EA'), symbol('EMC'), symbol('EMR'), symbol('ENDP'), symbol('ESV'), symbol('ETR'), symbol('EOG'), symbol('EQT'), symbol('EFX'), symbol('EQIX'), symbol('EQR'), symbol('ESS'), symbol('EL'), symbol('ES'), symbol('EXC'), symbol('EXPE'), symbol('EXPD'), symbol('ESRX'), symbol('XOM'), symbol('FFIV'), symbol('FB'), symbol('FAST'), symbol('FDX'), symbol('FIS'), symbol('FITB'), symbol('FSLR'), symbol('FE'), symbol('FISV'), symbol('FLIR'), symbol('FLS'), symbol('FLR'), symbol('FMC'), symbol('FTI'), symbol('F'), symbol('FOSL'), symbol('BEN'), symbol('FCX'), symbol('FTR'), symbol('GME'), symbol('GPS'), symbol('GRMN'), symbol('GD'), symbol('GE'), symbol('GGP'), symbol('GIS'), symbol('GM'), symbol('GPC'), symbol('GNW'), symbol('GILD'), symbol('GS'), symbol('GT'), symbol('GOOG'), symbol('GWW'), symbol('HAL'), symbol('HBI'), symbol('HOG'), symbol('HAR'), symbol('HRS'), symbol('HIG'), symbol('HAS'), symbol('HCA'), symbol('HCP'), symbol('HCN'), symbol('HP'), symbol('HES'), symbol('HPQ'), symbol('HD'), symbol('HON'), symbol('HRL'), symbol('HSP'), symbol('HST'), symbol('HCBK'), symbol('HUM'), symbol('HBAN'), symbol('ITW'), symbol('IR'), symbol('INTC'), symbol('ICE'), symbol('IBM'), symbol('IP'), symbol('IPG'), symbol('IFF'), symbol('INTU'), symbol('ISRG'), symbol('IVZ'), symbol('IRM'), symbol('JEC'), symbol('JBHT'), symbol('JNJ'), symbol('JCI'), symbol('JOY'), symbol('JPM'), symbol('JNPR'), symbol('KSU'), symbol('K'), symbol('KEY'), symbol('GMCR'), symbol('KMB'), symbol('KIM'), symbol('KMI'), symbol('KLAC'), symbol('KSS'), symbol('KRFT'), symbol('KR'), symbol('LB'), symbol('LLL'), symbol('LH'), symbol('LRCX'), symbol('LM'), symbol('LEG'), symbol('LEN'), symbol('LVLT'), symbol('LUK'), symbol('LLY'), symbol('LNC'), symbol('LLTC'), symbol('LMT'), symbol('L'), symbol('LOW'), symbol('LYB'), symbol('MTB'), symbol('MAC'), symbol('M'), symbol('MNK'), symbol('MRO'), symbol('MPC'), symbol('MAR'), symbol('MMC'), symbol('MLM'), symbol('MAS'), symbol('MA'), symbol('MAT'), symbol('MKC'), symbol('MCD'), symbol('MCK'), symbol('MJN'), symbol('MMV'), symbol('MDT'), symbol('MRK'), symbol('MET'), symbol('KORS'), symbol('MCHP'), symbol('MU'), symbol('MSFT'), symbol('MHK'), symbol('TAP'), symbol('MDLZ'), symbol('MON'), symbol('MNST'), symbol('MCO'), symbol('MS'), symbol('MOS'), symbol('MSI'), symbol('MUR'), symbol('MYL'), symbol('NDAQ'), symbol('NOV'), symbol('NAVI'), symbol('NTAP'), symbol('NFLX'), symbol('NWL'), symbol('NFX'), symbol('NEM'), symbol('NWSA'), symbol('NEE'), symbol('NLSN'), symbol('NKE'), symbol('NI'), symbol('NE'), symbol('NBL'), symbol('JWN'), symbol('NSC'), symbol('NTRS'), symbol('NOC'), symbol('NRG'), symbol('NUE'), symbol('NVDA'), symbol('ORLY'), symbol('OXY'), symbol('OMC'), symbol('OKE'), symbol('ORCL'), symbol('OI'), symbol('PCAR'), symbol('PLL'), symbol('PH'), symbol('PDCO'), symbol('PAYX'), symbol('PNR'), symbol('PBCT'), symbol('POM'), symbol('PEP'), symbol('PKI'), symbol('PRGO'), symbol('PFE'), symbol('PCG'), symbol('PM'), symbol('PSX'), symbol('PNW'), symbol('PXD'), symbol('PBI'), symbol('PCL'), symbol('PNC'), symbol('RL'), symbol('PPG'), symbol('PPL'), symbol('PX'), symbol('PCP'), symbol('PCLN'), symbol('PFG'), symbol('PG'), symbol('PGR'), symbol('PLD'), symbol('PRU'), symbol('PEG'), symbol('PSA'), symbol('PHM'), symbol('PVH'), symbol('PWR'), symbol('QCOM'), symbol('DGX'), symbol('RRC'), symbol('RTN'), symbol('O'), symbol('RHT'), symbol('REGN'), symbol('RF'), symbol('RSG'), symbol('RAI'), symbol('RHI'), symbol('ROK'), symbol('COL'), symbol('ROP'), symbol('ROST'), symbol('RLD'), symbol('R'), symbol('CRM'), symbol('SNDK'), symbol('SCG'), symbol('SLB'), symbol('SNI'), symbol('STX'), symbol('SEE'), symbol('SRE'), symbol('SHW'), symbol('SPG'), symbol('SWKS'), symbol('SLG'), symbol('SJM'), symbol('SNA'), symbol('SO'), symbol('LUV'), symbol('SWN'), symbol('SE'), symbol('STJ'), symbol('SWK'), symbol('SPLS'), symbol('SBUX'), symbol('HOT'), symbol('STT'), symbol('SRCL'), symbol('SYK'), symbol('STI'), symbol('SYMC'), symbol('SYY'), symbol('TROW'), symbol('TGT'), symbol('TEL'), symbol('TE'), symbol('THC'), symbol('TDC'), symbol('TSO'), symbol('TXN'), symbol('TXT'), symbol('HSY'), symbol('TRV'), symbol('TMO'), symbol('TIF'), symbol('TWX'), symbol('TWC'), symbol('TJX'), symbol('TMK'), symbol('TSS'), symbol('TSCO'), symbol('RIG'), symbol('TRIP'), symbol('FOXA'), symbol('TSN'), symbol('TYC'), symbol('UA'), symbol('UNP'), symbol('UNH'), symbol('UPS'), symbol('URI'), symbol('UTX'), symbol('UHS'), symbol('UNM'), symbol('URBN'), symbol('VFC'), symbol('VLO'), symbol('VAR'), symbol('VTR'), symbol('VRSN'), symbol('VZ'), symbol('VRTX'), symbol('VIAB'), symbol('V'), symbol('VNO'), symbol('VMC'), symbol('WMT'), symbol('WBA'), symbol('DIS'), symbol('WM'), symbol('WAT'), symbol('ANTM'), symbol('WFC'), symbol('WDC'), symbol('WU'), symbol('WY'), symbol('WHR'), symbol('WFM'), symbol('WMB'), symbol('WEC'), symbol('WYN'), symbol('WYNN'), symbol('XEL'), symbol('XRX'), symbol('XLNX'), symbol('XL'), symbol('XYL'), symbol('YHOO'), symbol('YUM'), symbol('ZION'), symbol('ZTS')]


    
    
  
    
    context.cons = ({'type': 'ineq', 'fun': constraint1},
                    {'type': 'ineq', 'fun': constraint2},
                    {'type': 'ineq', 'fun': constraint3})
    
    
    context.initial_theta = [1, 0.5, 0.5]
    

    context.avaliable_stocks = []
    
    pipe = Pipeline()  
    attach_pipeline(pipe, 'example')



    
    for i in range(len(stock_ids[:30])): #change back to :30
        temp_s = Stock(stock_ids[i])
        context.avaliable_stocks.append(temp_s)
    
 
    context.number_of_stocks = 0
    
   

    schedule_function(weekly_assessment, date_rules.every_day(), time_rules.market_open(hours=0,minutes=5))

def weekly_assessment(context,data):
    
  
    for s in context.avaliable_stocks:
        s.score = update_garch(s.ident,context,data)        
        

    t = context.avaliable_stocks
    temp_array  = sorted(t, key = by_score, reverse=True)
    temp_vix_array = sorted(t, key = by_score, reverse=True)
    context.number_of_stocks = 0
    scoreSum = 0.008
    
    
    for s in temp_array:
        
        if s.score>0.03 and context.number_of_stocks<10:
            scoreSum+= s.score       
    
    for s in temp_array:
        
        if not data.can_trade(s.ident):         
            continue
        
       
        if s.score>0 and context.number_of_stocks<3:
            order_target_percent(s.ident,s.score/scoreSum)
            print ("hello")
            s.low_price = data.current(s.ident,"price")*0.97 
            context.number_of_stocks+=1
        else:
            order_target_percent(s.ident,0)
      
    
    

                
def handle_data(context,data):
    
    lev = context.account.leverage
    record(l=lev)
                
    
def update_garch(equity, context, data):
    
    
    X=np.array(data.history(equity,'price',1000,'1d')[:-1])
    X=np.diff(np.log(X))
        

    objective = partial(negative_log_likelihood, X)


    result = scipy.optimize.minimize(objective, context.initial_theta, method='COBYLA', constraints = context.cons)
    
    
    theta_mle = result.x


    sigma_hats = np.sqrt(compute_squared_sigmas(X, np.sqrt(np.mean(X**2)), theta_mle))
    
    a0 = theta_mle[0]
    a1 = theta_mle[1] 
    b1 = theta_mle[2]
    sigma1 = sigma_hats[-1]
    

    current_stock_price=data.current(equity,"price")
    
   #Figure out how this works
    future_price = mc_simulate(current_stock_price, a0, a1, b1, sigma1)

    #Could do long-short here. If its less than 1 short it, if its greater than 1 long it. Or change universe to include low and high doing stocks
    
    
    
    return ((future_price/current_stock_price)-1)

  
def by_score(Stock):
      return Stock.score
      

def constraint1(theta):
    return 1 - (theta[1] + theta[2]) 

def constraint2(theta):
    return theta[1] 

def constraint3(theta):
    return theta[2] 

    
def negative_log_likelihood(X, theta):
       
    # Estimate initial sigma squared
    initial_sigma = np.sqrt(np.mean(X ** 2))
    
    # Generate the squared sigma values with the GARCH function
    sigma2 = compute_squared_sigmas(X, initial_sigma, theta)
    

    logL = -((-np.log(sigma2) - X**2/sigma2).sum())

    return logL

# We will use this function to forecast future values of the log return and sigma
def forecast_GARCH(T, a0, a1, b1, sigma1):
    
    # Initialize our values to hold log returns
    X = np.ndarray(T)
    
    #Setting up starting values
    sigma = np.ndarray(T)
    sigma[0] = sigma1
    
    for t in range(1, T):
        # Draw the next return
        X[t - 1] = sigma[t - 1] * np.random.normal(0, 1)
   
        # Draw the next sigma_t
        var_temp = a0 + b1 * sigma[t - 1]**2 + a1 * X[t - 1]**2
        if var_temp>=0:
               sigma[t] = math.sqrt(var_temp)
            
        
    
    #Last value
    X[T - 1] = sigma[T - 1] * np.random.normal(0, 1)    
    
    return X, sigma


# This function will help us estimate sigmas with the parameters according to the GARCH function
def compute_squared_sigmas(X, initial_sigma, theta):
    
    #Setting GARCH parameters
    a0 = theta[0]
    a1 = theta[1]
    b1 = theta[2]
    
    T = len(X)
    sigma2 = np.ndarray(T)
    
    sigma2[0] = initial_sigma ** 2
    
    for t in range(1, T):
        # Here's where we apply the GARCH equation
        sigma2[t] = a0 + a1 * X[t-1]**2 + b1 * sigma2[t-1]
    
    return sigma2


def mc_simulate(S0, a0, a1, b1, sigma1):
    price_forecast = []
    
    #Setting up time period for brownian motion
    T = 4
    dt = 1
    N = round(T/dt)
    
   
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt) 
    
    #Start of simulation
    for i in range(1,100):
        S = S0
        for j in range(0,T-1):
            change_forecast, sigma_forecast = forecast_GARCH(T, a0, a1, b1, sigma1) #forecasting 
             X = (change_forecast[j]-0.5*sigma_forecast[j]**2)*j + sigma_forecast[j]*W 
            S = S*np.exp(X) 
        
        price_forecast.append(S)
    
    #Return mean of price point probability distribution
    return np.mean(price_forecast)

#Keeping track of the stock
class Stock:
   def __init__(self, ident):
       self.sid = sid
       self.symbol = symbol
       self.ident = ident  
   
       self.low_price = 0