import numpy as np

# EE absolute increase between 43 and 31 dBm
ee_na_abs = 9.9297587719 - 9.1615452358
ee_oi_abs = 9.6498490727 - 9.1615452358
ee_cc_abs = 9.3297006876 - 9.1615452358
ee_ca_abs = 9.8077073486 - 9.1615452358

print(f'The highest inrease from reducing transmit power is {np.max([ee_na_abs, ee_oi_abs, ee_cc_abs])} from the non-adjacent scenario')


# Percentage reduction of EE between 34 and 4 dBm
ee_ca_pc = (9.8077073486 - 9.5734557084) / 9.8077073486
ee_na_pc = (9.9139017601 - 9.7941444724) / 9.9139017601 # non-adjacent, percentage-change
ee_oi_pc = (9.6427981924 - 9.55257456)   / 9.6427981924 # opposite inner
ee_cc_pc = (9.3297006876 - 9.2545405565) / 9.3297006876 # centre cell

print(ee_ca_pc)
print(ee_na_pc)
print(ee_oi_pc)
print(ee_cc_pc)


print(f'The highest percentage EE reduction is {np.max([ee_ca_pc, ee_na_pc, ee_oi_pc, ee_cc_pc])}')
print(f'The lowest percentage EE reduction is {np.min([ee_ca_pc, ee_na_pc, ee_oi_pc, ee_cc_pc])}')
print(f'The mean percentage EE reduction is {np.mean([ee_ca_pc, ee_na_pc, ee_oi_pc, ee_cc_pc])}')
print(f'The mean percentage reduction of EE for the lowest three is {np.mean([ee_na_pc, ee_oi_pc, ee_cc_pc])}') 


# Percentage change in EE between 43 and -inf dBm
sleep_ee_cc_pc = (9.5226937762 - 9.1615452358) / 9.5226937762   * 100     # centre cell
sleep_ee_oi_pc = (10.1389284518 - 9.1615452358) / 10.1389284518 * 100     # opposite inner
sleep_ee_na_pc = (10.7530557138 - 9.1615452358) / 10.7530557138 * 100     # non-adjacent
sleep_ee_ca_pc = (10.5083856357 - 9.1615452358) / 10.5083856357 * 100     # centre and adjacent

print('\n')
print(f'The highest percentage change in EE for SLEEP mode is {np.max([sleep_ee_cc_pc, sleep_ee_oi_pc, sleep_ee_na_pc, sleep_ee_ca_pc])} for the non-adjacent scenario')
print(f'The lowest percentage change in EE for SLEEP mode is {np.min([sleep_ee_cc_pc, sleep_ee_oi_pc, sleep_ee_na_pc, sleep_ee_ca_pc])} for the centre cell scenario')
print(f'The mean percentage change in EE for SLEEP mode is {np.mean([sleep_ee_cc_pc, sleep_ee_oi_pc, sleep_ee_na_pc, sleep_ee_ca_pc])}')




### Spectral efficiency
se_cc_pc = (1.9540090446 - 1.9241395316) / 1.9540090446 * 100
se_oi_pc = (1.9682909997 - 1.9348407167) / 1.9682909997 * 100
se_na_pc = (1.9773090335 - 1.9313283564) / 1.9773090335 * 100
se_ca_pc = (1.9651525574 - 1.8873838003) / 1.9651525574 * 100

print('\n')
print(f'Centre cell SE: {se_cc_pc}')
print(f'Opposite inner SE: {se_oi_pc}')
print(f'Non-adjacent SE: {se_na_pc}')
print(f'Centre and adjacent SE: {se_ca_pc}')
print('\n')
print(f'The highest percentage change in SE is {np.max([se_cc_pc, se_oi_pc, se_na_pc, se_ca_pc])} ')
print(f'The lowest percentage change in SE is {np.min([se_cc_pc, se_oi_pc, se_na_pc, se_ca_pc])} ')
print(f'The mean percentage change in SE is {np.mean([se_cc_pc, se_oi_pc, se_na_pc, se_ca_pc])} ')


# Centre cell highest: 
# Centre cell SE percentage change: 1.5286271618110379
# Opposite inner SE percentage change: 1.6994582104525429
# Non-adjacent SE percentage change: 2.325416832724959
# Centre and adjacent SE percentage change: 3.9573903210289187
#
# The highest percentage change in SE is 3.9573903210289187 
# The lowest percentage change in SE is 1.5286271618110379 
# The mean percentage change in SE is 2.3777231315043648