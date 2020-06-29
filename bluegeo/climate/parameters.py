def get_parameters():
    parameters = []
    for param in ['tmin', 'tmax', 'tave', 'rad', 'ppt', 'pas', 'eref']:
        for month in range(1, 13):
            if month < 10:
                month = '0{}'.format(month)

            parameters.append('{}{}'.format(param, month))
    return parameters


def get_scenarios():
    scenarios = []
    for year in range(1901, 2014):
        scenarios.append('normal_perioddat_year_{}'.format(year))
    # for decade in range(1900, 2010, 10):
    #     scenarios.append('normal_perioddat_decade_{}_{}'.format(decade + 1, decade + 10))
    # for year in range(2011, 2080):
    #     scenarios.append('gcm_gcmdat_annual_canesm2_rcp45_r1i1p1_{}'.format(year))
    # for year in range(2011, 2080):
    #     scenarios.append('gcm_gcmdat_annual_canesm2_rcp85_r1i1p1_{}'.format(year))
    return scenarios
