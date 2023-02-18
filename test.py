def main():
    periods = range(5, 20, 4)
    periods = [[range(periods[i], periods[i + 1] -1)] for i in range(len(periods) - 1)]
    periods.append([periods[-1][1] + 1, 20])
    print(periods)
if __name__ == "__main__":
    main()