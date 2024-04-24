const Button = (item) => {
  return (
    <a href={item.link} class="flex rounded-full bg-black text-white p-2 items-center gap-x-1 px-4">
      <item.iconType />
      <span>{item.name}</span>
    </a>
  );
};

const ButtonList = ({ links }) => {
  return (
    <div className="flex justify-center">
      {links.map((link) => (
        <div className="mx-1">
          <Button {...link} />
        </div>
        
      ))}
    </div>
  );
};

export default ButtonList;
