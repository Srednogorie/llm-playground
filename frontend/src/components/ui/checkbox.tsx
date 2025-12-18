import * as React from "react";

interface CheckboxProps {
  id: string;
  label: string;
  checked: boolean;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

const Checkbox = ({ id, label, checked, onChange }: CheckboxProps) => {
  return (
    <div className="flex items-center space-x-3 pl-2 pr-2 pt-1 pb-1 rounded-md hover:bg-gray-100 transition-colors cursor-pointer">
      <input
        type="checkbox"
        id={id}
        checked={checked}
        onChange={onChange}
        className="w-4 h-4 accent-blue-600 rounded cursor-pointer border-gray-300"
        value={label}
      />
      <label htmlFor={id} className="text-sm font-medium text-gray-700 cursor-pointer flex-1">
        {label}
      </label>
    </div>
  );
};

export default Checkbox;
