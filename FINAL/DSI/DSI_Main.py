import DSI.DSI_P4_HSW as HSW
import DSI.DSI_P35_OKC as OKC
import DSI.DSI_P1267_JHS as JHS
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('[ HSW ]')
    HSW.HSW_main()
    
    print('\n[ OKC ]')
    OKC.OKC_main()
    
    print('\n[ JHS ]')
    JHS.JHS_main()
    plt.show()