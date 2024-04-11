if __name__ == "__main__":
    args = parse_args()
    if args.disable_parallel:
        pandas_has_parallel = False
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    main(args)
